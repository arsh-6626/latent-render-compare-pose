"""SPEED+ test of a trained latent render-and-compare denoiser (liepose.exp.cmp.sp_dit) with best-of-N selection.

Per test image: N independent samples from the uniform SE(3) prior (sp_dit's sampler, 100-step DDIM, as in sp_dit.test),
then each final hypothesis is scored by the frozen world model's energy and the lowest-energy one is the estimate:
  reference = nearest *training* crop to the hypothesis pose (the sampler's bank query; nothing excluded at test)
  energy    = weighted_energy(WM(z_ref, g_ref -> g_hyp, crop meta), z_obs, FG_FLOOR + fg)
Inference reads only the test crop's tokens and box meta and the training crops through the bank. The GT pose is
read only by the scorer (esa_score.esa, full splits) and by two diagnostics that are reported separately and never
as the estimate: "sample0" (the first of the N samples = the 1-sample protocol of sp_dit.test, different seed) and
"oracle" (the best of the N by GT rotation error; an upper bound on what selection can reach).

  DATADIR=... PYTHONPATH=src python scripts/speedplus/test_lrc_bestofn.py \
      --exp "$LOGS/experiments/speedplus-lrc/train_steps=100000+seed=42+den_mode=dit+wm=sp_geo_v1" --step 100000 --n 16 \
      --out "$LOGS/speedplus_test/lrc_dit_sp_geo_v1_100k_best16"
"""

import argparse
import glob
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esa_score import allo_to_ego, esa  # noqa: E402

from liepose.data.speedplus.dataset import load_split, recover_translation  # noqa: E402
from liepose.exp.cmp.dit_ret import RetDenoiser  # noqa: E402
from liepose.exp.cmp.sp_dit import Testbed  # noqa: E402
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy  # noqa: E402
from liepose.exp.jepa_probe.probe_tless import TokenSplit  # noqa: E402
from liepose.exp.jepa_probe.sp_geometry import crop_meta  # noqa: E402
from liepose.utils.exploration import Options  # noqa: E402


def energy_fn(tb):
  params = {"params": tb.state.params_ema if tb.a.ema_enabled else tb.state.params}

  @jax.jit
  def e(zr, Rr, tr, kr, Rg, tg, kg, z_obs, w):
    z_hat = tb.model.apply(params, zr.astype(jnp.float32), Rr, tr, kr, Rg, tg, kg, method=RetDenoiser.render)
    return weighted_energy(z_hat, z_obs, w)

  return e


def rot_err(R, Rg):
  return np.degrees(np.arccos(np.clip((np.einsum("...ij,...ij->...", R, Rg) - 1) / 2, -1, 1)))


def run_split(tb, split, n, batch, out, limit=None):
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=batch, n_slices=1)
  D = os.environ["DATADIR"]
  obs = TokenSplit(os.path.join(D, "speedplus_tokens"), split)
  meta, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", split))
  assert (idx == obs.index).all() and np.abs(ts - obs.t).max() < 1e-5
  ann, _ = load_split(split)
  assert len(obs.obj) == len(ann["quat"]) and (np.sort(obs.index) == np.arange(len(ann["quat"]))).all(), "not the full split"
  box = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(os.path.join(D, "speedplus_crops", split, "shard_*.npz")))])
         for k in ["bbox", "bbox_center", "focal_ratio"]}
  lm, Lie = tb.lie_metrics, tb.Lie
  prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), tb.a.repr_type)))
  sample, poses, energy = tb.create_sampler(100), tb._poses_fn(), energy_fn(tb)
  key = jax.random.PRNGKey(tb.a.seed + 3)
  N = len(obs.obj) if limit is None else limit  # limit: smoke tests only (n is reported)
  R_all, t_all, E_all = np.zeros((N, n, 3, 3)), np.zeros((N, n, 3)), np.zeros((N, n))
  for s in range(0, N, batch):
    rows = np.arange(s, min(s + batch, N))
    rr = np.repeat(rows, n)
    tok, w, k = jnp.asarray(obs.tokens[rr], jnp.float32), jnp.asarray(FG_FLOOR + obs.fg[rr]), jnp.asarray(meta[rr])
    key, k1, k2 = jax.random.split(key, 3)
    rt = sample(k2, prior(jax.random.split(k1, len(rr))), tok, w, k)
    Rg, tg = map(np.asarray, poses(rt))
    zr, Rr, tr, kr = tb._refs(Rg, tg, np.full(len(rr), -1))
    E = np.asarray(energy(jnp.asarray(zr), jnp.asarray(Rr), jnp.asarray(tr), jnp.asarray(kr), jnp.asarray(Rg), jnp.asarray(tg),
                          k, tok, w))
    R_all[rows], t_all[rows], E_all[rows] = Rg.reshape(len(rows), n, 3, 3), tg.reshape(len(rows), n, 3), E.reshape(len(rows), n)
    if (s // batch) % 20 == 0:
      print(f"  {split} {rows[-1] + 1}/{N}", flush=True)

  i = obs.index[:N].astype(np.int64)
  box = {k: v[:N] for k, v in box.items()}
  R_gt = Rotation.from_quat(np.roll(ann["quat"][i], -1, -1)).as_matrix()
  t_gt = ann["tran"][i].astype(np.float64)

  def to_cam(R_allo, t_sc):
    t = recover_translation(t_sc, box["bbox"], box["bbox_center"], box["focal_ratio"], tb.a.tran_scale, tb.a.tran_offset)
    return allo_to_ego(R_allo, t), t

  Rc, tc = zip(*[to_cam(R_all[:, j], t_all[:, j]) for j in range(n)])
  Rc, tc = np.stack(Rc, 1), np.stack(tc, 1)  # (N, n, ...) camera frame
  ar = np.arange(N)
  pick = E_all.argmin(1)
  res = {"energy_pick": esa(Rc[ar, pick], tc[ar, pick], R_gt, t_gt)}  # the estimate
  err = rot_err(Rc, R_gt[:, None])
  res["diag_sample0"] = esa(Rc[:, 0], tc[:, 0], R_gt, t_gt)  # diagnostic: 1 sample
  orc = err.argmin(1)
  res["diag_oracle_rot"] = esa(Rc[ar, orc], tc[ar, orc], R_gt, t_gt)  # diagnostic: uses GT
  e_pick, e_0 = err[ar, pick], err[:, 0]
  res["diag_rot_share"] = {f"{m}_{k}": float(f(v)) for m, v in [("pick", e_pick), ("sample0", e_0), ("oracle", err[ar, orc])]
                           for k, f in [("lt5", lambda x: (x < 5).mean()), ("lt10", lambda x: (x < 10).mean()),
                                        ("gt90", lambda x: (x > 90).mean())]}
  res["diag_any_sample_lt10"] = float((err < 10).any(1).mean())
  for v in [v for v in res.values() if isinstance(v, dict) and "score" in v]:
    v["n"] = int(N)
  np.savez(os.path.join(out, f"{split}_hypotheses.npz"), index=i, R=Rc, t=tc, R_allo=R_all, t_scaled=t_all, energy=E_all)
  return res


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--exp", required=True)
  p.add_argument("--step", type=int, required=True)
  p.add_argument("--n", type=int, default=16)
  p.add_argument("--batch", type=int, default=25, help="images per batch (x n samples)")
  p.add_argument("--splits", nargs="+", default=["lightbox", "sunlamp"])
  p.add_argument("--out", required=True)
  p.add_argument("--limit", type=int, default=None, help="smoke tests only")
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  cfg = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(a.exp, "configs.yaml")), resolve=True))
  cfg.restore_path, cfg.restore_step, cfg.skip_train = None, a.step, True
  tb = Testbed(cfg, a.exp, setup=True)
  assert int(tb.state.step) == a.step, (int(tb.state.step), a.step)
  print(f"restored step {int(tb.state.step)}", flush=True)
  res = {"exp": a.exp, "step": a.step, "n": a.n, "limit": a.limit}
  for split in a.splits:
    res[split] = run_split(tb, split, a.n, a.batch, a.out, a.limit)
    print(split, json.dumps(res[split]), flush=True)
    json.dump(res, open(os.path.join(a.out, "esa.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
