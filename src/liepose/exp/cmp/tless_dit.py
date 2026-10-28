"""Latent render-and-compare SE(3) diffusion on T-LESS (dit_ret.RetDenoiser on LiePose's T-LESS crops), train + test.

The SPEED+ runner (sp_dit) on LiePose's T-LESS setup (liepose.exp.bop.run: SE(3), allocentric rotation +
translation_scaled, chordal loss on the noise, EMA, LR schedule, 100-step test sampling from the uniform prior,
final rt as the estimate). The denoiser sees the crop's frozen DINOv2 tokens (tless_tokens cache: LiePose's crops,
image x visible mask) and, at every step, the frozen world model's rendering (tless_ret_v1) of the current hypothesis
from the nearest training crop of the same object (TlessBank, allocentric R + symmetric equivalents as the world
model was trained; training: the observation's own source image excluded).

  - Object identity: the target's obj_id (given by the BOP task) selects the reference bank; there is no other
    object conditioning, as in LiePose.
  - No crop conditioning (the T-LESS caches hold no box meta); the WM is geometry-free (tless_ret_v1), so the crop
    meta arguments are zeros. LiePose's ResNet likewise sees only the crop.
  - Translations enter the world model divided by its T_SCALE (as in its training), the denoiser's SE(3) input
    unchanged.

mode "dit_nowm": z_hat = 0 (same denoiser and inputs without the world model; the control).

modes "dit_ft" / "dit_nowm_ft" (trained backbone, as sp_dit.py): LiePose's own T-LESS image pipeline (crops with its DZI /
colour augmentation, image x visible mask, as the token cache was encoded) replaces the token cache; the denoiser runs
DINOv2 itself: a frozen copy gives z_obs in the world model's token space (2x2-pooled), so the comparison is unchanged,
and a trainable copy (lr x bb_lr_mult) gives its 16 x 16 tokens as an extra stream; the token weights come from the
crop's visible mask as encode_crops.py computes them (8 x 8 area mean, 3 x 3 max). Test crops: LiePose's TestDataset.

train_objs (held-out objects, zero-shot test): the denoiser trains on these objects' crops only; the reference bank
keeps every object's training crops (an unseen object's posed reference views, never trained on), and the world
model should be trained on the same objects (probe_ret_tless --train_objs). The denoiser has no object input, so on
an unseen object everything it knows about that object comes through the world model's references. Score seen /
unseen with scripts/tless/score_subsets.py.

Test: LiePose's own TestDataset (tless-test_primesense, the bop19 target images) gives each instance's index; its
tokens come from the test cache by that index (asserted to match); estimates go through LiePose's Evaluator
(_process_pose, BOP CSV, bop_toolkit AR on test_targets_bop19). GT is read only by bop_toolkit.

  DATADIR=... BOP_RENDERER=vispy python -m liepose.exp.cmp.tless_dit --root_path $LOGS/experiments/tless-lrc \
      "den_mode=[dit]" "wm_dir=[$LOGS/probe/tless_ret_v1]" "train_steps=[100000]"
"""

import json
import os
import time
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze, unfreeze
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from liepose.exp.bop import run as bop_run
from liepose.exp.jepa_probe import probe_ret_tless as prt
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.probe_tless import TokenSplit, load_symmetries
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Options

from .dit_ret import RetDenoiser

MODES = ("dit", "dit_nowm", "dit_in2", "dit_in3", "dit_hi", "dit_lo", "dit_ft", "dit_nowm_ft")  # dit_ret.py (+ _ft)
FT = ("dit_ft", "dit_nowm_ft")


def mask_fg(mask):
  """(N, 224, 224) visible mask -> (N, 64) token weights, as encode_crops.py (8 x 8 area mean, then 3 x 3 max)"""
  m = mask.reshape(mask.shape[0], 8, 28, 8, 28).mean((2, 4))
  m = jax.lax.reduce_window(m, -jnp.inf, jax.lax.max, (1, 3, 3), (1, 1, 1), ((0, 0), (1, 1), (1, 1)))
  return m.reshape(mask.shape[0], 64)
K_DIM = 3  # placeholder crop meta (unused by a geometry-free WM; constant input to k_emb)


def no_crop(k):
  return jnp.zeros(k.shape[:-1] + (1,), k.dtype)


class TokenTrainData:
  """Random training crops from the token cache; yields LiePose-style batches."""

  def __init__(self, split, batch_size, seed, objs=None):
    self.split, self.batch_size, self.rng = split, batch_size, np.random.default_rng(seed)
    self.rows = np.arange(len(split.obj)) if not objs else np.flatnonzero(np.isin(split.obj, objs))
    if objs:
      print(f"train_objs: {len(self.rows)} of {len(split.obj)} training crops ({sorted(objs)})", flush=True)

  def __iter__(self):
    return self

  def __next__(self):
    b = self.rows[self.rng.integers(0, len(self.rows), self.batch_size)]
    q = Rotation.from_matrix(self.split.R[b]).as_quat()  # xyzw
    return {"row": b, "image": np.zeros((len(b), 1, 1, 3), np.float32), "mask": np.ones((len(b), 1, 1, 1), np.float32),
            "rotation_allo": np.roll(q, 1, -1).astype(np.float32), "translation_scaled": self.split.t[b].astype(np.float32)}


class Testbed(bop_run.TestbedSE3):

  def __init__(self, a, exp_path, setup=True):
    assert a.den_mode in MODES, a.den_mode
    self.ft = a.den_mode in FT
    super().__init__(a, exp_path, setup=setup)

  # --- model + caches ---

  def create_model(self):
    wa = SimpleNamespace(**json.load(open(os.path.join(self.a.wm_dir, "args.json"))))
    assert not getattr(wa, "geo", False) and wa.models_info is None, "expects the T-LESS probe_ret_tless model"
    self.wa = wa
    if self.a.train_objs:  # held-out objects: the world model must not have trained on the unseen objects either
      assert sorted(getattr(wa, "train_objs", None) or []) == sorted(self.a.train_objs), "world model saw other objects"
    self.t_scale = np.asarray(wa.T_SCALE, np.float32)
    prt.T_SCALE = self.t_scale.astype(np.float64)  # the bank's key normalisation (module global)
    D = os.environ["DATADIR"]
    self.cache = os.path.join(D, "tless_tokens")
    self.train_split = TokenSplit(self.cache, "train")
    syms = load_symmetries(os.path.join(D, "bop_datasets/tless/models_eval/models_info.json"))
    self.bank = prt.TlessBank(self.train_split, np.arange(len(self.train_split.obj)), syms, wa.lam, wa.max_sym)
    model = RetDenoiser(mode=self.a.den_mode.replace("_ft", ""), gate_t0=self.a.timesteps // 2, dim=self.a.dit_dim,
                        depth=self.a.dit_depth, wm_dim=wa.dim, wm_depth=wa.depth, wm_geo=False, crop_fn=no_crop, ft=self.ft)
    n = 2
    z, e, v3, k = jnp.zeros((n, 64, 384)), jnp.tile(jnp.eye(3), (n, 1, 1)), jnp.zeros((n, 3)), jnp.zeros((n, K_DIM))
    params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 6)), e, v3, jnp.zeros((n, 1)), z, e, v3,
                                 k, z, jnp.ones((n, 64)), k, image=jnp.zeros((n, 224, 224, 3)) if self.ft else None))
    src = flax.serialization.msgpack_restore(open(os.path.join(self.a.wm_dir, "params.msgpack"), "rb").read())["params"]
    shapes = lambda p: jax.tree_util.tree_map(np.shape, p)
    assert shapes(src) == shapes(params["params"]["wm"]), "world model checkpoint does not match RetPoseModel"
    params["params"]["wm"] = jax.tree_util.tree_map(jnp.asarray, src)
    if self.ft:
      dino = flax.serialization.msgpack_restore(open(os.path.join(D, "dinov2_vits14_224.msgpack"), "rb").read())["params"]
      for name in ("dino", "dino_frozen"):
        assert shapes(dino) == shapes(params["params"][name]), "DINOv2 checkpoint does not match"
        params["params"][name] = jax.tree_util.tree_map(jnp.asarray, dino)
    params["batch_stats"] = {}
    return model, FrozenDict(params)

  def create_optimizer_and_lr(self):
    optim, lr = super().create_optimizer_and_lr()
    if not self.ft:
      return optim, lr
    # trainable backbone at lr x bb_lr_mult; the frozen DINOv2 copy and the world model get no update at all
    bb = optax.adamw(learning_rate=lambda s: self.a.bb_lr_mult * lr(s))
    label = lambda path, _: {"dino": "bb", "dino_frozen": "frozen", "wm": "frozen"}.get(path[0].key, "main")
    labels = lambda p: jax.tree_util.tree_map_with_path(label, p)
    return optax.multi_transform({"main": optim, "bb": bb, "frozen": optax.set_to_zero()}, labels), lr

  def _poses_fn(self):
    lm = self.lie_metrics

    @jax.jit
    def poses(rt):
      g = jax.vmap(lm.as_lie)(rt)
      return jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g)

    return poses

  def _refs(self, obj, R, t, exclude):
    """nearest training crop of the same object at the (noisy) poses -> tokens, R, t / T_SCALE"""
    img, Rr, tr = self.bank.query(obj, np.asarray(R, np.float64), np.asarray(t, np.float64), exclude)
    return self.train_split.tokens[img], Rr.astype(np.float32), (tr / self.t_scale).astype(np.float32)

  # --- train ---

  def create_train_dataset(self):
    if self.ft:  # LiePose's own image pipeline (augmented crops, visible masks)
      assert not self.a.train_objs, "train_objs is for the token-cache modes"
      return super().create_train_dataset()
    return TokenTrainData(self.train_split, self.a.batch_size, self.a.seed, self.a.train_objs)

  def create_get_flat_batch_train_fn(self):
    base = super().create_get_flat_batch_train_fn()  # LiePose noising of (rotation_allo, translation_scaled)
    poses = self._poses_fn()
    d = self.train_split

    def get_flat_batch_train(seed, batch, n_slices=256):
      if self.ft:  # images go to the backbones once per image (masked as the cache), not once per slice
        batch = dict(batch)
        mask = np.asarray(batch["mask"], np.float32)
        image = np.asarray(batch["image"], np.float32) * mask
        batch["image"], batch["mask"] = np.zeros((len(image), 1, 1, 3), np.float32), np.ones((len(image), 1, 1, 1), np.float32)
        obj, src = np.asarray(batch["object_id"]), np.asarray(batch["index"])
      else:
        row = batch["row"]
        obj, src = d.obj[row], d.index[row]
      b = base(seed, batch, n_slices=n_slices)
      b.pop("img", None)
      s = b["rt"].shape[0] // len(src)
      Rg, tg = map(np.asarray, poses(b["rt"]))
      zr, Rr, tr = self._refs(np.repeat(obj, s), Rg, tg, np.repeat(src, s))
      b.update(Rg=jnp.asarray(Rg), tg=jnp.asarray(tg / self.t_scale), zr=jnp.asarray(zr), Rr=jnp.asarray(Rr), tr=jnp.asarray(tr))
      if self.ft:
        b.update(image=jnp.asarray(image), fg=mask_fg(jnp.asarray(mask[..., 0])))
      else:
        b.update(tok=jnp.asarray(d.tokens[row]), fg=jnp.asarray(d.fg[row]))
      return b

    return get_flat_batch_train

  def create_update_fn(self):
    lie_metrics, loss_name, model = self.lie_metrics, self.a.loss_name, self.model
    assert self.a.learn_noise

    def update_fn(state, batch):
      def loss_fn(params):
        t = batch["t"].reshape((-1, 1)).astype(jnp.float32)
        s = batch["rt"].shape[0] // batch["fg"].shape[0]
        z_ext = None
        if "image" in batch:
          z_obs, z_ext = model.apply({"params": params}, batch["image"], method=RetDenoiser.encode)
          z_ext = jnp.repeat(z_ext, s, 0)
        else:
          z_obs = batch["tok"].astype(jnp.float32)
        z_obs = jnp.repeat(z_obs, s, 0)
        w = jnp.repeat(FG_FLOOR + batch["fg"], s, 0)
        k = jnp.zeros((batch["rt"].shape[0], K_DIM))
        mu = model.apply({"params": params}, batch["rt"], batch["Rg"], batch["tg"], t, batch["zr"].astype(jnp.float32),
                         batch["Rr"], batch["tr"], k, z_obs, w, k, None, z_ext, method=RetDenoiser.noise)
        return jnp.mean(jax.vmap(lie_metrics.get_distance_fn(loss_name))(batch["zt"], mu))

      loss, grads = jax.value_and_grad(loss_fn)(state.params)
      new_state = state.apply_gradients(grads=grads)
      new_state = new_state.replace(params=freeze({**unfreeze(new_state.params), "wm": state.params["wm"]}))
      return loss, new_state

    return update_fn

  # --- test ---

  def create_sampler(self, n_steps=100):
    model, poses = self.model, self._poses_fn()
    params = {"params": self.state.params_ema if self.a.ema_enabled else self.state.params}
    p_fn = jax.vmap(self.create_p_sample_fn())
    T = self.noise_schedule.timesteps
    cur = (np.linspace(T, 0, n_steps, endpoint=False) - 1).astype(np.int32).tolist()  # = Testbed.run_test
    seq = list(zip(cur, cur[1:] + [0]))
    t_scale = jnp.asarray(self.t_scale)

    @jax.jit
    def step(key, rt, t, tp, Rg, tg, zr, Rr, tr, z_obs, w, z_ext):
      B = rt.shape[0]
      key, k = jax.random.split(key)
      kz = jnp.zeros((B, K_DIM))
      mu = model.apply(params, rt, Rg, tg / t_scale, jnp.full((B, 1), t, jnp.float32), zr.astype(jnp.float32), Rr, tr, kz,
                       z_obs, w, kz, None, z_ext, method=RetDenoiser.noise)
      _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      return key, rt

    encode = jax.jit(lambda image: model.apply(params, image, method=RetDenoiser.encode))

    def sample(key, rt, obj, tok, fg, image=None):
      """tok: the cached tokens, or (ft) None with the masked crop images, encoded once per image"""
      z_ext = None
      if image is not None:
        tok, z_ext = encode(image)
      for t, tp in seq:
        Rg, tg = poses(rt)
        zr, Rr, tr = self._refs(obj, np.asarray(Rg), np.asarray(tg), np.full(len(rt), -1))
        key, rt = step(key, rt, jnp.int32(t), jnp.int32(tp), Rg, tg, zr, Rr, tr, tok, fg, z_ext)
      return rt

    return sample

  def run_test(self, opt, opt_path, limit=None):
    """LiePose's TestDataset + Evaluator; tokens by dataset index; 1 sample from the uniform prior per instance.
    limit: smoke tests only."""
    self.opt, self.opt_path = opt, opt_path
    os.makedirs(opt_path, exist_ok=True)
    obs = TokenSplit(self.cache, "test")
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
      if self.ft:  # the test crop itself (image x visible mask, as the cache), weights from its mask
        mask = np.asarray(batch["mask"], np.float32)
        image = jnp.asarray(np.asarray(batch["image"], np.float32) * mask)
        rt = sample(k2, prior(jax.random.split(k1, len(rows))), obs.obj[rows], None, FG_FLOOR + mask_fg(jnp.asarray(mask[..., 0])), image)
      else:
        rt = sample(k2, prior(jax.random.split(k1, len(rows))), obs.obj[rows],
                    jnp.asarray(obs.tokens[rows], jnp.float32), jnp.asarray(FG_FLOOR + obs.fg[rows]))
      rt = jax.block_until_ready(rt)
      dt = time.time() - t0
      seq_rt = self.post_process(rt)[None, :, None]  # (time=1, b, slices=1, 7) wxyz_xyz, final sample
      evaluator.process(batch, {"seq_r0": seq_rt, "seq_rt": seq_rt, "time": dt})
      n += len(rows)
      if limit is not None and n >= limit:
        break
    evaluator.summarize()
    return n


def build_explorations():
  e = bop_run.build_explorations()
  e.den_mode = ["dit"]
  e.dit_dim = [256]
  e.dit_depth = [4]
  e.wm_dir = [None]
  e.train_steps = [100000]
  e.batch_size = [32]
  e.n_slices = [16]
  e.ckpt_steps = [25000]
  e.test_limit = [None]
  e.train_objs = [None]
  e.bb_lr_mult = [0.1]  # *_ft: backbone learning rate relative to the denoiser's  # held-out-object test: train on these objects only (see the module docstring)
  return e


SAMPLE = dict(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=100, n_slices=1)


def main():
  a = get_args(exp_name="tless-lrc", desc="Latent render-and-compare diffusion on T-LESS")
  for opt in load_config_from_args(a, build_explorations()).expand():
    opt.pop("sample_options", None)
    wm = os.path.basename(os.path.normpath(opt.wm_dir))
    objs = f"+objs={min(opt.train_objs)}-{max(opt.train_objs)}" if opt.train_objs else ""
    exp_path = os.path.join(a.root_path, opt.to_string(includes=["den_mode", "train_steps", "seed"]) + f"+wm={wm}{objs}")
    tb = Testbed(opt, exp_path, setup=True)
    if not opt.skip_train:
      save_config(opt, os.path.join(exp_path, "configs.yaml"))
      tb.run_train()
    step = int(tb.state.step)
    out = os.path.join(exp_path, f"test_{step}")
    n = tb.run_test(Options(SAMPLE), out, limit=opt.test_limit)
    tb.run_eval(Options(SAMPLE), out)
    json.dump({"exp": exp_path, "step": step, "n_instances": n, "sample": SAMPLE,
               "renderer": os.environ.get("BOP_RENDERER", "cpp")}, open(os.path.join(out, "run.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
