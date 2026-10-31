"""T-LESS SE(3) diffusion with the feature-field world model (tless_field.TlessField) in the loop: sp_field_dit's
denoiser on tless_dit's T-LESS setup (LiePose's crops, BOP19 test, Evaluator).

At every denoising step the frozen field renders the current hypothesis of the target object (obj_id from the BOP
task) in the observation's own crop camera; the denoiser compares it with the observed tokens (sp_field_dit modes).
The crop meta (box + intrinsics, tless_field.crop_meta10) enters the render only: the denoiser has no crop
conditioning (use_k=False), as tless_dit and LiePose, so field vs retrieval world model differ in the world model alone.

  token-cache modes  crop meta per cache row from {cache}/{split}_crop_meta.npy (tless_field.tless_crop_meta of
                     export_crops.py --meta shards aligned with the cache)
  *_ft               LiePose's live image pipeline (tless_dit dit_ft): meta from the batch, image x visible mask

  DATADIR=... BOP_RENDERER=vispy python -m liepose.exp.cmp.tless_field_dit --root_path $LOGS/experiments/tless-field \
      "den_mode=[field_full_lat]" "field_dir=[$LOGS/probe/tl_field_v1]" "train_steps=[100000]"
"""

import json
import os
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, unfreeze
from jaxlie import SO3
from tqdm import tqdm

from liepose.exp.bop import run as bop_run
from liepose.exp.bop.ds_config import DS
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.tless_field import crop_meta10, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Options

from . import tless_dit
from .sp_field_dit import BASE_MODES, FieldDenoiser, FtFieldDenoiser
from .tless_dit import mask_fg

MODES = BASE_MODES + tuple(m + "_ft" for m in BASE_MODES)
SAMPLE = tless_dit.SAMPLE


class Testbed(tless_dit.Testbed):

  def __init__(self, a, exp_path, setup=True):
    assert a.den_mode in MODES, a.den_mode
    self.ft = a.den_mode.endswith("_ft")
    self.base_mode = a.den_mode[:-3] if self.ft else a.den_mode
    bop_run.TestbedSE3.__init__(self, a, exp_path, setup=setup)

  def create_model(self):
    D = os.environ["DATADIR"]
    self.cache = self.a.tl_cache or os.path.join(D, DS["tokens"])
    self.train_split = TokenSplit(self.cache, "train")
    if not self.ft:
      self.train_meta = np.load(os.path.join(self.cache, "train_crop_meta.npy")).astype(np.float32)
      assert len(self.train_meta) == len(self.train_split.obj)
    self.field_model, self.field_params = load_tless_field(self.a.field_dir)
    n, z = 2, jnp.zeros((2, 64, 384))
    hz = jnp.zeros((n, 64, self.field_model.hid))
    k2 = jnp.zeros((n, 10))
    if self.ft:
      model = FtFieldDenoiser(mode=self.base_mode, dim=self.a.dit_dim, depth=self.a.dit_depth, use_k=False)
      params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 224, 224, 3)), jnp.zeros((n, 6)), jnp.zeros((n, 1)), k2,
                                   jnp.ones((n, 64)), jnp.ones((n, 64)), hz, method=FtFieldDenoiser.full_init))
      dino = flax.serialization.msgpack_restore(open(os.path.join(D, "dinov2_vits14_224.msgpack"), "rb").read())["params"]
      shapes = lambda p: jax.tree_util.tree_map(np.shape, p)
      for name in ("dino", "dino_frozen"):
        assert shapes(dino) == shapes(params["params"][name]), "DINOv2 checkpoint does not match"
        params["params"][name] = jax.tree_util.tree_map(jnp.asarray, dino)
    else:
      model = FieldDenoiser(mode=self.base_mode, dim=self.a.dit_dim, depth=self.a.dit_depth, use_k=False)
      params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 6)), jnp.zeros((n, 1)), k2,
                                   z, jnp.ones((n, 64)), z, jnp.ones((n, 64)), hz))
    params["batch_stats"] = {}
    return model, FrozenDict(params)

  def _render(self, obj, Rg, tg, kg, z_obs):
    """frozen field of each hypothesis' object at (allocentric R, translation_scaled) in the observation's crop"""
    return jax.lax.stop_gradient(self.field_model.apply(self.field_params, obj, Rg, tg, kg, app_stats(z_obs), with_latent=True))

  # --- train ---

  def create_sample_train_fn(self):
    """process=fm: flow matching on SE(3) instead of LiePose's diffusion. The pose moves on the geodesic from the data
    pose g0 (tau=0) to a prior pose g1 (tau=1, the sampler's prior: uniform rotation, translation in [-1, 1]^3),
    g_tau = g0 Exp(tau xi), xi = Log(g0^-1 g1); the network predicts the constant body-frame velocity xi. t = tau * T
    keeps the step embedding on the diffusion's scale."""
    if self.a.process == "diffusion":
      return super().create_sample_train_fn()
    assert self.a.process == "fm", self.a.process
    Lie, lm, T, rep = self.Lie, self.lie_metrics, self.noise_schedule.timesteps, self.a.repr_type

    def sample_train_fn(seed, rot, tran):
      k1, k2 = jax.random.split(seed)
      tau = jax.random.uniform(k2, ())
      g0 = Lie.from_rotation_and_translation(SO3(rot), tran)
      xi = (g0.inverse() @ Lie.sample_uniform(k1)).log()
      gt = g0 @ Lie.exp(tau * xi)
      return {"rt": lm.as_repr(gt, rep), "t": tau * T, "zt": xi, "r0": lm.as_repr(g0, rep)}

    return sample_train_fn

  def create_get_flat_batch_train_fn(self):
    base = bop_run.TestbedSE3.create_get_flat_batch_train_fn(self)  # LiePose noising of (rotation_allo, translation_scaled)
    poses = self._poses_fn()
    d = self.train_split

    def get_flat_batch_train(seed, batch, n_slices=256):
      if self.ft:  # images go to the backbones once per image (masked as the cache), not once per slice
        batch = dict(batch)
        mask = np.asarray(batch["mask"], np.float32)
        image = np.asarray(batch["image"], np.float32) * mask
        meta = crop_meta10({k: np.asarray(batch[k]) for k in ("bbox", "bbox_center", "dzi_size", "focal_ratio", "cam")})
        obj = np.asarray(batch["object_id"])
        batch["image"], batch["mask"] = np.zeros((len(image), 1, 1, 3), np.float32), np.ones((len(image), 1, 1, 1), np.float32)
      else:
        row = batch["row"]
        obj, meta = d.obj[row], self.train_meta[row]
      b = base(seed, batch, n_slices=n_slices)
      b.pop("img", None)
      s = b["rt"].shape[0] // len(obj)
      Rg, tg = poses(b["rt"])
      b.update(Rg=Rg, tg=tg, kg=jnp.asarray(np.repeat(meta, s, 0)), obj=jnp.asarray(np.repeat(obj, s)))
      if self.ft:
        b.update(image=jnp.asarray(image), fg=mask_fg(jnp.asarray(mask[..., 0])))
      else:
        b.update(tok=jnp.asarray(d.tokens[row]), fg=jnp.asarray(d.fg[row]))
      return b

    return get_flat_batch_train

  def create_update_fn(self):
    lie_metrics, loss_name, model = self.lie_metrics, self.a.loss_name, self.model
    assert self.a.learn_noise
    fm = self.a.process == "fm"

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
      z_hat, opa, h = self._render(mb["obj"], mb["Rg"], mb["tg"], mb["kg"], z_obs)
      mu = model.apply({"params": params}, mb["rt"], t, mb["kg"], z_obs, w, z_hat, opa, h, keep, z_ext)
      if fm:  # velocity regression in se(3)
        return jnp.mean(jnp.sum((mu - mb["zt"]) ** 2, -1))
      return jnp.mean(jax.vmap(lie_metrics.get_distance_fn(loss_name))(mb["zt"], mu))

    def update_fn(state, batch):
      keys = ("rt", "t", "zt", "Rg", "tg", "kg", "obj", "fg") + tuple(k for k in ("tok", "image") if k in batch)
      mb = {k: batch[k] for k in keys}
      n_img = mb["fg"].shape[0]
      if self.base_mode == "field_drop":  # per image (all its slices), from the step's noise draws (no extra rng state)
        u = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(0), state.step), (n_img,))
        mb["keep"] = (u >= self.a.drop_p).astype(jnp.float32)
      m = self.a.grad_accum
      if m == 1:
        loss, grads = jax.value_and_grad(loss_fn)(state.params, mb)
      else:  # the same mean loss over m image-major chunks, sequentially (sp_field_dit.update_fn)
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

  # --- test ---

  def create_sampler(self, n_steps=100):
    model, poses = self.model, self._poses_fn()
    params = {"params": self.state.params_ema if self.a.ema_enabled else self.state.params}
    T = self.noise_schedule.timesteps
    fm = self.a.process == "fm"
    if fm:  # Euler steps of the ODE on the group, tau 1 -> 0: g <- g Exp(-(tau - tau') v)
      lm, rep = self.lie_metrics, self.a.repr_type
      taus = np.linspace(1.0, 0.0, n_steps + 1)
      seq = [(float(a) * T, float(a - b)) for a, b in zip(taus[:-1], taus[1:])]
      move = jax.vmap(lambda r, v, d: lm.as_repr(lm.as_lie(r) @ self.Lie.exp(-d * v), rep), (0, 0, None))
    else:
      p_fn = jax.vmap(self.create_p_sample_fn())
      cur = (np.linspace(T, 0, n_steps, endpoint=False) - 1).astype(np.int32).tolist()  # = Testbed.run_test
      seq = list(zip(cur, cur[1:] + [0]))

    @jax.jit
    def step(key, rt, t, tp, obj, z_obs, w, kg, z_ext):
      """diffusion: t, tp = step and previous step; fm: t = tau * T, tp = the step size in tau"""
      B = rt.shape[0]
      key, k = jax.random.split(key)
      Rg, tg = poses(rt)
      z_hat, opa, h = self._render(obj, Rg, tg, kg, z_obs)
      mu = model.apply(params, rt, jnp.full((B, 1), t, jnp.float32), kg, z_obs, w, z_hat, opa, h, None, z_ext)
      if fm:
        return key, move(rt, mu, tp)
      _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      return key, rt

    encode = jax.jit(lambda image: model.apply(params, image, method=FtFieldDenoiser.encode)) if self.ft else None

    def sample(key, rt, obj, tok, fg, meta, image=None):
      """tok: the cached tokens, or (ft) None with the masked crop images, encoded once per image"""
      z_ext = None
      if image is not None:
        tok, z_ext = encode(image)
      obj = jnp.asarray(obj)
      for t, tp in seq:
        t, tp = (jnp.float32(t), jnp.float32(tp)) if fm else (jnp.int32(t), jnp.int32(tp))
        key, rt = step(key, rt, t, tp, obj, tok, fg, meta, z_ext)
      return rt

    return sample

  def run_test(self, opt, opt_path, limit=None):
    """tless_dit.run_test with the crop meta (the test cache's rows, or the batch's for *_ft)"""
    self.opt, self.opt_path = opt, opt_path
    os.makedirs(opt_path, exist_ok=True)
    obs = TokenSplit(self.cache, "test")
    test_meta = None if self.ft else np.load(os.path.join(self.cache, "test_crop_meta.npy")).astype(np.float32)
    row_of = {int(i): r for r, i in enumerate(obs.index)}
    assert len(row_of) == len(obs.index)
    dataset = self.create_test_dataset()
    assert len(dataset.dataset) == len(obs.index), (len(dataset.dataset), len(obs.index))
    evaluator = self.create_test_evaluator(dataset)
    lm, Lie = self.lie_metrics, self.Lie
    prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), self.a.repr_type)))
    sample = self.create_sampler(opt.steps)
    key, n = jax.random.PRNGKey(self.a.seed + 2), 0
    for batch in tqdm(dataset, desc="Sample"):
      rows = np.array([row_of[int(i)] for i in np.asarray(batch["index"])])
      assert (obs.obj[rows] == np.asarray(batch["object_id"])).all(), "token cache misaligned with the test dataset"
      key, k1, k2 = jax.random.split(key, 3)
      t0 = time.time()
      if self.ft:
        mask = np.asarray(batch["mask"], np.float32)
        image = jnp.asarray(np.asarray(batch["image"], np.float32) * mask)
        meta = crop_meta10({k: np.asarray(batch[k]) for k in ("bbox", "bbox_center", "dzi_size", "focal_ratio", "cam")})
        rt = sample(k2, prior(jax.random.split(k1, len(rows))), obs.obj[rows], None, FG_FLOOR + mask_fg(jnp.asarray(mask[..., 0])),
                    jnp.asarray(meta), image)
      else:
        rt = sample(k2, prior(jax.random.split(k1, len(rows))), obs.obj[rows], jnp.asarray(obs.tokens[rows], jnp.float32),
                    jnp.asarray(FG_FLOOR + obs.fg[rows]), jnp.asarray(test_meta[rows]))
      rt = jax.block_until_ready(rt)
      dt = time.time() - t0
      seq_rt = self.post_process(rt)[None, :, None]
      evaluator.process(batch, {"seq_r0": seq_rt, "seq_rt": seq_rt, "time": dt})
      n += len(rows)
      if limit is not None and n >= limit:
        break
    evaluator.summarize()
    return n


def build_explorations():
  e = tless_dit.build_explorations()
  e.den_mode = ["field_full_lat"]
  e.field_dir = [None]
  e.drop_p = [0.5]
  e.tl_cache = [None]  # token cache with {split}_crop_meta.npy (default $DATADIR/tless_tokens)
  e.grad_accum = [1]  # micro-batches per step (same gradient, less memory; sp_field_dit)
  e.process = ["diffusion"]  # diffusion (LiePose's) | fm (flow matching on SE(3), create_sample_train_fn)
  e.fm_steps = ["10/20/100"]  # fm: ODE steps of the test runs (one BOP19 test each)
  return e


def main():
  a = get_args(exp_name="tless-field", desc="Feature-field world model in T-LESS SE(3) diffusion")
  for opt in load_config_from_args(a, build_explorations()).expand():
    opt.pop("sample_options", None)
    field = os.path.basename(os.path.normpath(opt.field_dir))
    exp_path = os.path.join(a.root_path, opt.to_string(includes=["den_mode", "train_steps", "seed"]) + f"+field={field}"
                            + ("" if opt.process == "diffusion" else f"+process={opt.process}"))
    tb = Testbed(opt, exp_path, setup=True)
    if not opt.skip_train:
      save_config(opt, os.path.join(exp_path, "configs.yaml"))
      tb.run_train()
    step = int(tb.state.step)
    runs = [(f"test_{step}", SAMPLE)] if opt.process == "diffusion" else \
           [(f"test_{step}_n{n}", dict(SAMPLE, steps=int(n))) for n in str(opt.fm_steps).split("/")]
    for name, sample in runs:
      out = os.path.join(exp_path, name)
      n = tb.run_test(Options(sample), out, limit=opt.test_limit)
      tb.run_eval(Options(sample), out)
      json.dump({"exp": exp_path, "step": step, "n_instances": n, "sample": sample, "process": opt.process,
                 "renderer": os.environ.get("BOP_RENDERER", "cpp")}, open(os.path.join(out, "run.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
