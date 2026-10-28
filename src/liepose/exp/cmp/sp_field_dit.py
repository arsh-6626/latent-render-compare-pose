"""SPEED+ latent render-and-compare diffusion with the feature-field world model, and an information bottleneck.

LiePose's SPEED+ diffusion (as sp_dit.py), with the world model replaced by the frozen 3D feature field
(liepose.exp.jepa_probe.field: no retrieval, no reference images; it renders a pose hypothesis in the observation's
crop from its own learned 3D model of the object). Every denoising step, for every hypothesis g_t:
  render   z_hat(g_t), opacity(g_t), h(g_t) = field(g_t, crop, appearance code of the observation)    (frozen)
  denoise  tokens below + conditioning (t, g_t, crop) -> adaLN token transformer -> noise -> SE(3) step
den_mode (what the denoiser sees per token):
  field_full        [z_obs, z_hat, z_hat - z_obs, w]                       (the current LaRC input, field world model)
  field_bneck       comparison only, never the observation's features: cos(z_hat_i, z_obs_i), |z_hat_i - z_obs_i|^2,
                    opacity_i and a 3x3 local correlation cos(z_obs_i, z_hat_j), j the 9 rendered tokens around i.
                    Information bottleneck: any evidence about the pose reaches the denoiser only through the
                    comparison of the observation with the world model's render.
  field_bneck_lat   field_bneck + h(g_t): the field's own latent of the surface point each ray hits (pooled to
                    tokens, never decoded into coordinates / depth): the world model adds its 3D knowledge in
                    latent space; h depends on the hypothesis only, not on the observation.
  field_nowm        [z_obs, w] only (the control: the same denoiser and budget without the world model)
  field_full_lat    field_full + h(g_t) (the world model's latent added to the full input)
  field_drop        soft bottleneck: field_full's raw channels [z_obs, z_hat, z_hat - z_obs, w] plus field_bneck's
                    comparison signals; in training the raw channels of a sample are zeroed with probability
                    --drop_p, so the denoiser must learn to pose from the comparison alone too; never at test
  *_ft              any of the above with the backbone trained (as sp_dit.py dit_ft): LiePose's own augmented image crops
                    instead of the token cache; a frozen DINOv2 copy gives z_obs (the field's comparison space, so the
                    world model is unchanged) and a trainable copy (lr x bb_lr_mult) adds its 16 x 16 tokens (2 x 2 cells
                    merged into channels, projected) to the denoiser's tokens; test crops from the crop shards' images.
Results (frozen DINOv2, 100k, official ESA lightbox / sunlamp): field_full 0.742 / 0.687, field_nowm 0.827 / 0.743,
field_bneck 1.516 / 1.507, field_bneck_lat 1.318 / 1.326.
Training / test / scoring as sp_dit.py (token caches, official ESA on lightbox / sunlamp). Inference reads the test
crop's tokens and box meta and the frozen field; GT only in the scorer.

  DATADIR=... SPEEDPLUS_ROOT=... python -m liepose.exp.cmp.sp_field_dit --root_path $LOGS/speedplus-field "den_mode=[field_bneck]" \
      "field_dir=[$LOGS/probe/sp_field_v1]" "train_steps=[100000]"
"""

import json
import os

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

from liepose.exp.jepa_probe.comparator import load_field
from liepose.exp.jepa_probe.probe import AdaLNBlock
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import crop_feat, crop_k, crop_meta
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.exp.model import PosEmbed
from liepose.models.dinov2 import DinoV2, pool2
from liepose.exp.speedplus import run as sp_run
from liepose.utils.config import get_args, load_config_from_args, save_config

from . import sp_dit

BASE_MODES = ("field_full", "field_bneck", "field_bneck_lat", "field_nowm", "field_full_lat", "field_drop")
MODES = BASE_MODES + tuple(m + "_ft" for m in BASE_MODES)


def local_corr(z_obs, z_hat, grid=8):
  """(N, P, D) -> (N, P, 9): cosine of each observed token with the rendered tokens in its 3x3 neighbourhood
  (zero outside the grid)"""
  N, P, D = z_obs.shape
  n = lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
  o, h = n(z_obs).reshape(N, grid, grid, D), n(z_hat).reshape(N, grid, grid, D)
  hp = jnp.pad(h, ((0, 0), (1, 1), (1, 1), (0, 0)))
  out = [(o * hp[:, 1 + dy:1 + dy + grid, 1 + dx:1 + dx + grid]).sum(-1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
  return jnp.stack(out, -1).reshape(N, P, 9)


class FieldDenoiser(nn.Module):
  mode: str = "field_bneck"
  dim: int = 256
  depth: int = 4
  heads: int = 8
  use_k: bool = True  # False (T-LESS): no crop conditioning, the crop meta enters the render only (as LiePose / tless_dit)

  @nn.compact
  def __call__(self, rt, t, k, z_obs, w, z_hat, opa, h, keep=None, z_ext=None):
    """rt (N, 6) noisy pose, t (N, 1), k (N, 6) crop meta, z_obs (N, 64, D), w (N, 64), the field's z_hat (N, 64, D),
    opa (N, 64), h (N, 64, H) -> noise (N, 6)"""
    ln = lambda name, x: nn.LayerNorm(name=name)(x)
    cmp = jnp.concatenate([((z_hat * z_obs).sum(-1) / (jnp.linalg.norm(z_hat, axis=-1) * jnp.linalg.norm(z_obs, axis=-1) + 1e-6))[..., None],
                           ((z_hat - z_obs) ** 2).mean(-1, keepdims=True), opa[..., None], local_corr(z_obs, z_hat)], -1)
    full = lambda: jnp.concatenate([ln("ln_obs", z_obs), ln("ln_hat", z_hat), ln("ln_diff", z_hat - z_obs), w[..., None]], -1)
    if self.mode == "field_full":
      x = full()
    elif self.mode == "field_full_lat":
      x = jnp.concatenate([full(), ln("ln_lat", h)], -1)
    elif self.mode == "field_drop":  # keep (N,) in {0, 1}: 0 = the raw channels dropped (training only)
      raw = full() * (jnp.ones(z_obs.shape[:1]) if keep is None else keep)[:, None, None]
      x = jnp.concatenate([raw, cmp], -1)
    elif self.mode == "field_nowm":
      x = jnp.concatenate([ln("ln_obs", z_obs), w[..., None]], -1)
    elif self.mode == "field_bneck":
      x = cmp
    else:  # field_bneck_lat
      x = jnp.concatenate([cmp, ln("ln_lat", h)], -1)
    x = nn.Dense(self.dim, name="tok_in")(x) + self.param("pos", nn.initializers.normal(0.02), (1, x.shape[1], self.dim))
    if z_ext is not None:  # *_ft: the trainable backbone's tokens (already projected to dim)
      x = x + z_ext
    c = PosEmbed(256, self.dim, True, name="t_emb")(t) + PosEmbed(256, self.dim, False, name="g_emb")(rt)
    if self.use_k:
      c = c + nn.Dense(self.dim, name="k_emb")(crop_feat(k))
    c = nn.Dense(self.dim, name="c_mlp")(jax.nn.silu(c))
    x = jnp.concatenate([c[:, None], x], 1)
    for i in range(self.depth):
      x = AdaLNBlock(self.dim, self.heads, name=f"block_{i}")(x, c)
    return nn.Dense(6, name="noise_out")(nn.LayerNorm()(x[:, 0]))


class FtFieldDenoiser(nn.Module):
  """*_ft: FieldDenoiser + a frozen DINOv2 copy (z_obs, the field's comparison space) + a trainable copy (z_ext)"""
  mode: str = "field_full_lat"
  dim: int = 256
  depth: int = 4
  use_k: bool = True

  def setup(self):
    self.den = FieldDenoiser(mode=self.mode, dim=self.dim, depth=self.depth, use_k=self.use_k)
    self.dino_frozen = DinoV2()
    self.dino = DinoV2()
    self.ext_ln = nn.LayerNorm()
    self.ext_proj = nn.Dense(self.dim)

  def encode(self, image):
    """image (n, 224, 224, 3) 0-255 -> z_obs (n, 64, 384) frozen 2x2-pooled tokens, z_ext (n, 64, dim)"""
    z_obs = pool2(jax.lax.stop_gradient(self.dino_frozen(image)))
    z = self.dino(image)
    n, _, D = z.shape
    z_ext = z.reshape(n, 8, 2, 8, 2, D).transpose(0, 1, 3, 2, 4, 5).reshape(n, 64, 4 * D)
    return z_obs, self.ext_proj(self.ext_ln(z_ext))

  def __call__(self, rt, t, k, z_obs, w, z_hat, opa, h, keep=None, z_ext=None):
    return self.den(rt, t, k, z_obs, w, z_hat, opa, h, keep, z_ext)

  def full_init(self, image, rt, t, k, w, opa, h):  # touches every submodule
    z_obs, z_ext = self.encode(image)
    return self.den(rt, t, k, z_obs, w, z_obs, opa, h, None, z_ext)


class Testbed(sp_dit.Testbed):

  def __init__(self, a, exp_path, setup=True):
    assert a.den_mode in MODES, a.den_mode
    self.ft = a.den_mode.endswith("_ft")
    self.base_mode = a.den_mode[:-3] if self.ft else a.den_mode
    sp_run.TestbedSE3.__init__(self, a, exp_path, setup=setup)

  def create_model(self):
    D = os.environ["DATADIR"]
    self.train_split = TokenSplit(os.path.join(D, "speedplus_tokens"), "train")
    self.train_meta, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", "train"))
    assert (idx == self.train_split.index).all() and np.abs(ts - self.train_split.t).max() < 1e-5
    self.field_model, self.field_params = load_field(self.a.field_dir)
    n, z = 2, jnp.zeros((2, 64, 384))
    hz = jnp.zeros((n, 64, self.field_model.field.hid))
    k2 = jnp.asarray(self.train_meta[:n])
    if self.ft:
      model = FtFieldDenoiser(mode=self.base_mode, dim=self.a.dit_dim, depth=self.a.dit_depth)
      params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 224, 224, 3)), jnp.zeros((n, 6)), jnp.zeros((n, 1)), k2,
                                   jnp.ones((n, 64)), jnp.ones((n, 64)), hz, method=FtFieldDenoiser.full_init))
      dino = flax.serialization.msgpack_restore(open(os.path.join(D, "dinov2_vits14_224.msgpack"), "rb").read())["params"]
      shapes = lambda p: jax.tree_util.tree_map(np.shape, p)
      for name in ("dino", "dino_frozen"):
        assert shapes(dino) == shapes(params["params"][name]), "DINOv2 checkpoint does not match"
        params["params"][name] = jax.tree_util.tree_map(jnp.asarray, dino)
    else:
      model = FieldDenoiser(mode=self.base_mode, dim=self.a.dit_dim, depth=self.a.dit_depth)
      params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 6)), jnp.zeros((n, 1)), k2,
                                   z, jnp.ones((n, 64)), z, jnp.ones((n, 64)), hz))
    params["batch_stats"] = {}
    return model, FrozenDict(params)

  def _render(self, Rg, tg, kg, z_obs):
    """frozen field at the hypotheses (allocentric R, translation_scaled) in the observation's crop"""
    return jax.lax.stop_gradient(self.field_model.apply(self.field_params, None, None, None, None, Rg, tg, kg, app_stats(z_obs),
                                                        with_latent=True))

  def create_train_dataset(self):
    if self.ft:  # LiePose's image pipeline (sp_dit.Testbed's ft branch)
      return super().create_train_dataset()
    return sp_dit.TokenTrainData(self.train_split, self.a.batch_size, self.a.seed)

  def create_get_flat_batch_train_fn(self):
    base = sp_run.TestbedSE3.create_get_flat_batch_train_fn(self)
    poses = self._poses_fn()
    d = self.train_split

    def get_flat_batch_train(seed, batch, n_slices=256):
      if self.ft:  # images go to the backbones once per image, not once per slice
        batch = dict(batch)
        image = np.asarray(batch["image"], np.float32)
        batch["image"] = np.zeros((len(image), 1, 1, 3), np.float32)
        meta = crop_k({k: np.asarray(batch[k]) for k in ("bbox", "bbox_center", "dzi_size", "focal_ratio")}).astype(np.float32)
      else:
        row = batch["row"]
        meta = self.train_meta[row]
      b = base(seed, batch, n_slices=n_slices)
      b.pop("img", None)
      s = b["rt"].shape[0] // len(meta)
      Rg, tg = poses(b["rt"])
      b.update(Rg=Rg, tg=tg, kg=jnp.asarray(np.repeat(meta, s, 0)))
      if self.ft:
        b.update(image=jnp.asarray(image), fg=jnp.ones((len(meta), 64), jnp.float32))  # SPEED+: uniform (as the caches)
      else:
        b.update(tok=jnp.asarray(d.tokens[row]), fg=jnp.asarray(d.fg[row]))
      return b

    return get_flat_batch_train

  def create_update_fn(self):
    lie_metrics, loss_name, model = self.lie_metrics, self.a.loss_name, self.model
    assert self.a.learn_noise

    def loss_fn(params, mb):
      """mb: (part of) a batch; per-image keys (fg, tok / image, keep) and per-slice keys, slices image-major"""
      t = mb["t"].reshape((-1, 1)).astype(jnp.float32)
      s = mb["rt"].shape[0] // mb["fg"].shape[0]
      w = jnp.repeat(FG_FLOOR + mb["fg"], s, 0)
      keep = jnp.repeat(mb["keep"], s, 0) if "keep" in mb else None
      z_ext = None
      if "image" in mb:  # frozen copy -> z_obs (the field's comparison space), trainable copy -> z_ext
        z_obs, z_ext = model.apply({"params": params}, mb["image"], method=FtFieldDenoiser.encode)
        z_ext = jnp.repeat(z_ext, s, 0)
      else:
        z_obs = mb["tok"].astype(jnp.float32)
      z_obs = jnp.repeat(z_obs, s, 0)
      z_hat, opa, h = self._render(mb["Rg"], mb["tg"], mb["kg"], z_obs)
      mu = model.apply({"params": params}, mb["rt"], t, mb["kg"], z_obs, w, z_hat, opa, h, keep, z_ext)
      return jnp.mean(jax.vmap(lie_metrics.get_distance_fn(loss_name))(mb["zt"], mu))

    def update_fn(state, batch):
      keys = ("rt", "t", "zt", "Rg", "tg", "kg", "fg") + tuple(k for k in ("tok", "image") if k in batch)
      mb = {k: batch[k] for k in keys}
      n_img = mb["fg"].shape[0]
      if self.base_mode == "field_drop":  # per image (all its slices), from the step's noise draws (no extra rng state)
        u = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(0), state.step), (n_img,))
        mb["keep"] = (u >= self.a.drop_p).astype(jnp.float32)
      m = self.a.grad_accum
      if m == 1:
        loss, grads = jax.value_and_grad(loss_fn)(state.params, mb)
      else:  # the same mean loss over m image-major chunks, sequentially (the full batch's gradient in less memory)
        assert n_img % m == 0, (n_img, m)
        chunks = jax.tree_util.tree_map(lambda x: x.reshape((m, x.shape[0] // m) + x.shape[1:]), mb)

        def acc(carry, c):
          l, g = jax.value_and_grad(loss_fn)(state.params, c)
          return jax.tree_util.tree_map(jnp.add, carry, (l, g)), None

        zero = (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, state.params))
        (loss, grads), _ = jax.lax.scan(acc, zero, chunks)
        loss, grads = jax.tree_util.tree_map(lambda x: x / m, (loss, grads))
      return loss, state.apply_gradients(grads=grads)

    return update_fn

  def create_sampler(self, n_steps=100):
    model, poses = self.model, self._poses_fn()
    params = {"params": self.state.params_ema if self.a.ema_enabled else self.state.params}
    p_fn = jax.vmap(self.create_p_sample_fn())
    T = self.noise_schedule.timesteps
    cur = (np.linspace(T, 0, n_steps, endpoint=False) - 1).astype(np.int32).tolist()
    seq = list(zip(cur, cur[1:] + [0]))

    @jax.jit
    def step(key, rt, t, tp, z_obs, w, kg, z_ext):
      B = rt.shape[0]
      key, k = jax.random.split(key)
      Rg, tg = poses(rt)
      z_hat, opa, h = self._render(Rg, tg, kg, z_obs)
      mu = model.apply(params, rt, jnp.full((B, 1), t, jnp.float32), kg, z_obs, w, z_hat, opa, h, None, z_ext)
      _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      return key, rt

    encode = jax.jit(lambda image: model.apply(params, image, method=FtFieldDenoiser.encode)) if self.ft else None

    def sample(key, rt, tok, fg, meta, image=None):
      """tok: the cached tokens, or (ft) None with the crop images, encoded once per image"""
      z_ext = None
      if image is not None:
        tok, z_ext = encode(image)
      for t, tp in seq:
        key, rt = step(key, rt, jnp.int32(t), jnp.int32(tp), tok, fg, meta, z_ext)
      return rt

    return sample


def build_explorations():
  e = sp_dit.build_explorations()
  e.den_mode = ["field_bneck"]
  e.field_dir = [None]
  e.drop_p = [0.5]  # field_drop: probability of dropping a training image's raw channels
  e.grad_accum = [1]  # micro-batches per step (same gradient, less memory: 64 x 64 on 40 GB GPUs)
  return e


def main():
  a = get_args(exp_name="speedplus-field", desc="Field world-model render-and-compare diffusion on SPEED+")
  for opt in load_config_from_args(a, build_explorations()).expand():
    opt.pop("sample_options", None)
    fld = os.path.basename(os.path.normpath(opt.field_dir))
    exp_path = os.path.join(a.root_path, opt.to_string(includes=["den_mode", "train_steps", "seed"]) + f"+field={fld}")
    tb = Testbed(opt, exp_path, setup=True)
    if not opt.skip_train:
      save_config(opt, os.path.join(exp_path, "configs.yaml"))
      tb.run_train()
    step = int(tb.state.step)
    out = os.path.join(exp_path, f"test_{step}")
    res = {"exp": exp_path, "step": step}
    for split in opt.test_splits:
      res[split] = sp_dit.test(tb, split, out, limit=opt.test_limit)
      print(split, json.dumps(res[split]), flush=True)
    os.makedirs(out, exist_ok=True)
    name = "esa.json" if list(opt.test_splits) == ["lightbox", "sunlamp"] else f"esa_{'_'.join(opt.test_splits)}.json"
    json.dump(res, open(os.path.join(out, name), "w"), indent=2)


if __name__ == "__main__":
  main()
