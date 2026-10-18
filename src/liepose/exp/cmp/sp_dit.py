"""Latent render-and-compare SE(3) diffusion on SPEED+ (dit_ret.RetDenoiser on SPEED+ crops), train + test.

Everything about the diffusion is LiePose's SPEED+ setup (liepose.exp.speedplus.run: SE(3), allocentric rotation +
translation_scaled, power noise schedule, chordal loss on the noise, EMA, LR schedule, 100-step test sampling from
the uniform prior, final rt as the estimate). What changes is the denoiser: instead of a ResNet on the image, the
dit token transformer on the crop's frozen DINOv2 tokens, given at every step the frozen world model's rendering
of the current hypothesis (probe_ret_tless model: sp_ret_v1, or sp_geo_v1 with geometry tokens).

  observation  -> crop (keypoint box, as LiePose) -> DINOv2 tokens z_obs (speedplus_tokens cache), box meta k_obs
  each step, each sample at its noisy pose g_t:
    reference  = nearest training crop to g_t (TlessBank: allocentric R; training: the observation's own source
                 image excluded; test: nothing excluded, the bank holds training crops only)
    render     z_hat = WM(z_ref, g_ref -> g_t) in the observation's crop                         (frozen)
    compare    [z_obs, z_hat, z_hat - z_obs, w] -> adaLN transformer conditioned on (t, g_t, k_obs) -> noise

mode "dit_nowm": z_hat = 0 (same denoiser and inputs without the world model; the control).

Inference reads only the test crop's tokens and box meta (and the training crops through the bank); the GT pose
is read by the scorer only (scripts/speedplus/esa_score.esa on the full lightbox / sunlamp splits).

  DATADIR=... python -m liepose.exp.cmp.sp_dit --root_path $LOGS/experiments/speedplus-lrc "den_mode=[dit]" \
      "wm_dir=[$LOGS/probe/sp_ret_v1]" "train_steps=[100000]"
"""

import json
import os
import sys
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze
from scipy.spatial.transform import Rotation

from liepose.data.speedplus.dataset import load_split, recover_translation
from liepose.exp.jepa_probe import probe_ret_tless as prt
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import SpeedPlusGeometry, crop_feat, crop_meta
from liepose.exp.speedplus import run as sp_run
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Options

from .dit_ret import RetDenoiser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../scripts/speedplus"))
from esa_score import allo_to_ego, esa  # noqa: E402

MODES = ("dit", "dit_nowm")


class TokenTrainData:
  """Random training crops (both augmented copies) from the token cache; yields LiePose-style batches."""

  def __init__(self, split, batch_size, seed):
    self.split, self.batch_size, self.rng = split, batch_size, np.random.default_rng(seed)

  def __iter__(self):
    return self

  def __next__(self):
    b = self.rng.integers(0, len(self.split.obj), self.batch_size)
    q = Rotation.from_matrix(self.split.R[b]).as_quat()  # xyzw
    return {"row": b, "image": np.zeros((len(b), 1, 1, 3), np.float32), "rotation_allo": np.roll(q, 1, -1).astype(np.float32),
            "translation_scaled": self.split.t[b].astype(np.float32)}


class Testbed(sp_run.TestbedSE3):

  def __init__(self, a, exp_path, setup=True):
    assert a.den_mode in MODES, a.den_mode
    super().__init__(a, exp_path, setup=setup)

  # --- model + caches ---

  def create_model(self):
    wa = SimpleNamespace(**json.load(open(os.path.join(self.a.wm_dir, "args.json"))))
    assert list(wa.T_SCALE) == [1.0, 1.0, 1.0] and wa.models_info == "none", "expects a SPEED+ probe_ret_tless model"
    self.wa = wa
    prt.T_SCALE = np.asarray(wa.T_SCALE)  # the bank's key normalisation (module global in probe_ret_tless)
    D = os.environ["DATADIR"]
    self.train_split = TokenSplit(os.path.join(D, "speedplus_tokens"), "train")
    self.train_meta, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", "train"))
    assert (idx == self.train_split.index).all() and np.abs(ts - self.train_split.t).max() < 1e-5
    syms = {int(o): np.eye(3)[None] for o in np.unique(self.train_split.obj)}
    self.obj_id = int(self.train_split.obj[0])
    self.bank = prt.TlessBank(self.train_split, np.arange(len(self.train_split.obj)), syms, wa.lam)
    geo = bool(getattr(wa, "geo", False))
    model = RetDenoiser(mode=self.a.den_mode, dim=self.a.dit_dim, depth=self.a.dit_depth, wm_dim=wa.dim,
                        wm_depth=wa.depth, wm_geo=geo, geo_fn=SpeedPlusGeometry() if geo else None, crop_fn=crop_feat)
    n = 2
    z, e, v3 = jnp.zeros((n, 64, 384)), jnp.tile(jnp.eye(3), (n, 1, 1)), jnp.zeros((n, 3))
    k = jnp.asarray(self.train_meta[:n])
    params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 6)), e, v3, jnp.zeros((n, 1)), z, e, v3,
                                 k, z, jnp.ones((n, 64)), k))
    src = flax.serialization.msgpack_restore(open(os.path.join(self.a.wm_dir, "params.msgpack"), "rb").read())["params"]
    shapes = lambda p: jax.tree_util.tree_map(np.shape, p)
    assert shapes(src) == shapes(params["params"]["wm"]), "world model checkpoint does not match RetPoseModel"
    params["params"]["wm"] = jax.tree_util.tree_map(jnp.asarray, src)
    params["batch_stats"] = {}
    return model, FrozenDict(params)

  def _poses_fn(self):
    lm = self.lie_metrics

    @jax.jit
    def poses(rt):
      g = jax.vmap(lm.as_lie)(rt)
      return jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g)

    return poses

  def _refs(self, R, t, exclude):
    img, Rr, tr = self.bank.query(np.full(len(R), self.obj_id), np.asarray(R, np.float64), np.asarray(t, np.float64), exclude)
    return self.train_split.tokens[img], Rr.astype(np.float32), tr.astype(np.float32), self.train_meta[img]

  # --- train ---

  def create_train_dataset(self):
    return TokenTrainData(self.train_split, self.a.batch_size, self.a.seed)

  def create_get_flat_batch_train_fn(self):
    base = super().create_get_flat_batch_train_fn()  # LiePose noising of (rotation_allo, translation_scaled)
    poses = self._poses_fn()
    d = self.train_split

    def get_flat_batch_train(seed, batch, n_slices=256):
      b = base(seed, batch, n_slices=n_slices)
      b.pop("img", None)
      row = batch["row"]
      s = b["rt"].shape[0] // len(row)
      Rg, tg = map(np.asarray, poses(b["rt"]))
      zr, Rr, tr, kr = self._refs(Rg, tg, np.repeat(d.index[row], s))
      b.update(Rg=jnp.asarray(Rg), tg=jnp.asarray(tg), zr=jnp.asarray(zr), Rr=jnp.asarray(Rr), tr=jnp.asarray(tr),
               kr=jnp.asarray(kr), kg=jnp.asarray(np.repeat(self.train_meta[row], s, 0)),
               tok=jnp.asarray(d.tokens[row]), fg=jnp.asarray(d.fg[row]))
      return b

    return get_flat_batch_train

  def create_update_fn(self):
    lie_metrics, loss_name, model = self.lie_metrics, self.a.loss_name, self.model
    assert self.a.learn_noise

    def update_fn(state, batch):
      def loss_fn(params):
        t = batch["t"].reshape((-1, 1)).astype(jnp.float32)
        s = batch["rt"].shape[0] // batch["tok"].shape[0]
        z_obs = jnp.repeat(batch["tok"].astype(jnp.float32), s, 0)
        w = jnp.repeat(FG_FLOOR + batch["fg"], s, 0)
        mu = model.apply({"params": params}, batch["rt"], batch["Rg"], batch["tg"], t, batch["zr"].astype(jnp.float32),
                         batch["Rr"], batch["tr"], batch["kr"], z_obs, w, batch["kg"], method=RetDenoiser.noise)
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

    @jax.jit
    def step(key, rt, t, tp, Rg, tg, zr, Rr, tr, kr, z_obs, w, kg):
      B = rt.shape[0]
      key, k = jax.random.split(key)
      mu = model.apply(params, rt, Rg, tg, jnp.full((B, 1), t, jnp.float32), zr.astype(jnp.float32), Rr, tr, kr, z_obs, w,
                       kg, method=RetDenoiser.noise)
      _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      return key, rt

    def sample(key, rt, tok, fg, meta):
      for t, tp in seq:
        Rg, tg = poses(rt)
        zr, Rr, tr, kr = self._refs(np.asarray(Rg), np.asarray(tg), np.full(len(rt), -1))
        key, rt = step(key, rt, jnp.int32(t), jnp.int32(tp), Rg, tg, zr, Rr, tr, kr, tok, fg, meta)
      return rt

    return sample


def test(tb, split, out_dir, batch=100, steps=100, limit=None):
  """Full split, 1 sample per image from the uniform SE(3) prior; GT only in the score."""
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=steps, batch_size=batch, n_slices=1)
  D = os.environ["DATADIR"]
  obs = TokenSplit(os.path.join(D, "speedplus_tokens"), split)
  meta, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", split))
  assert (idx == obs.index).all() and np.abs(ts - obs.t).max() < 1e-5
  ann, _ = load_split(split)
  assert len(obs.obj) == len(ann["quat"]) and (np.sort(obs.index) == np.arange(len(ann["quat"]))).all(), "not the full split"
  box = {k: np.concatenate([np.load(f)[k] for f in sorted(__import__("glob").glob(os.path.join(D, "speedplus_crops", split, "shard_*.npz")))])
         for k in ["bbox", "bbox_center", "focal_ratio"]}
  n_obs = len(obs.obj) if limit is None else min(limit, len(obs.obj))  # limit: smoke tests only (n is reported)
  lm, Lie = tb.lie_metrics, tb.Lie
  prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), tb.a.repr_type)))
  sample, poses = tb.create_sampler(steps), tb._poses_fn()
  key = jax.random.PRNGKey(tb.a.seed + 2)
  R_allo, t_sc = [], []
  for s in range(0, n_obs, batch):
    rows = np.arange(s, min(s + batch, n_obs))
    key, k1, k2 = jax.random.split(key, 3)
    rt = sample(k2, prior(jax.random.split(k1, len(rows))), jnp.asarray(obs.tokens[rows], jnp.float32),
                jnp.asarray(FG_FLOOR + obs.fg[rows]), jnp.asarray(meta[rows]))
    Rg, tg = poses(rt)
    R_allo.append(np.asarray(Rg, np.float64))
    t_sc.append(np.asarray(tg, np.float64))
    if (s // batch) % 10 == 0:
      print(f"  {split} {rows[-1] + 1}/{n_obs}", flush=True)
  R_allo, t_sc = np.concatenate(R_allo), np.concatenate(t_sc)
  box = {k: v[:n_obs] for k, v in box.items()}
  t_est = recover_translation(t_sc, box["bbox"], box["bbox_center"], box["focal_ratio"], tb.a.tran_scale, tb.a.tran_offset)
  R_est = allo_to_ego(R_allo, t_est)
  i = obs.index[:n_obs].astype(np.int64)
  R_gt = Rotation.from_quat(np.roll(ann["quat"][i], -1, -1)).as_matrix()
  r = esa(R_est, t_est, R_gt, ann["tran"][i].astype(np.float64))
  r["n"] = int(len(i))
  os.makedirs(out_dir, exist_ok=True)
  np.savez(os.path.join(out_dir, f"{split}_estimates.npz"), index=i, R=R_est, t=t_est, R_allo=R_allo, t_scaled=t_sc)
  return r


def build_explorations():
  e = sp_run.build_explorations()
  e.den_mode = ["dit"]
  e.dit_dim = [256]
  e.dit_depth = [4]
  e.wm_dir = [None]
  e.train_steps = [100000]
  e.batch_size = [32]
  e.n_slices = [16]
  e.ckpt_steps = [25000]
  e.test_splits = [["lightbox", "sunlamp"]]
  e.test_limit = [None]
  return e


def main():
  a = get_args(exp_name="speedplus-lrc", desc="Latent render-and-compare diffusion on SPEED+")
  for opt in load_config_from_args(a, build_explorations()).expand():
    opt.pop("sample_options", None)
    wm = os.path.basename(os.path.normpath(opt.wm_dir))
    exp_path = os.path.join(a.root_path, opt.to_string(includes=["den_mode", "train_steps", "seed"]) + f"+wm={wm}")
    tb = Testbed(opt, exp_path, setup=True)
    if not opt.skip_train:
      save_config(opt, os.path.join(exp_path, "configs.yaml"))
      tb.run_train()
    step = int(tb.state.step)
    out = os.path.join(exp_path, f"test_{step}")
    res = {"exp": exp_path, "step": step}
    for split in opt.test_splits:
      res[split] = test(tb, split, out, limit=opt.test_limit)
      print(split, json.dumps(res[split]), flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "esa.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
