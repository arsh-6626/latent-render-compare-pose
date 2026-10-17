"""Energy as a verifier: rerank K LiePose samples per image with the v3 latent energy.

For every image, compare the error of (a) a random sample (= baseline), (b) the lowest-energy
sample, (c) the best sample (oracle ceiling).

  PYTHONPATH=src python scripts/verifier_eval.py --samples errs_k16.npz --cache $DATADIR/dino_cache \
      --probe $DATADIR/logs/probe/v3_soft15_full
"""

import argparse
import json
import os

import flax
import jax
import jax.numpy as jnp
import numpy as np

from liepose.exp.jepa_probe.probe import pose_feat
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, FrozenPoseModel, pool_tokens, weighted_energy

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--samples", required=True)
  p.add_argument("--cache", required=True)
  p.add_argument("--probe", required=True)
  p.add_argument("--chunk", type=int, default=2048)
  a = p.parse_args()

  s = np.load(a.samples)
  K = int(s["n_slices"])
  R, t, idx, lab = s["R"], s["t"], s["index"], s["label"]
  rot_err, tran_err = s["rot_err"], s["tran_err"]
  N = len(idx) // K
  assert (idx.reshape(N, K) == idx.reshape(N, K)[:, :1]).all(), "samples of one image must be contiguous"

  with open(os.path.join(a.probe, "args.json")) as f:
    pa = json.load(f)
  meta = np.load(os.path.join(a.cache, "test-5k_meta.npz"))
  assert (meta["label"][idx] == lab).all(), "token cache is not aligned with dataset indices"
  img_idx = idx.reshape(N, K)[:, 0]
  tok = np.load(os.path.join(a.cache, "test-5k_tokens.npy"), mmap_mode="r")
  fg = np.load(os.path.join(a.cache, "test-5k_fg.npy"), mmap_mode="r")
  order = np.argsort(img_idx)
  tok_i, fg_i = pool_tokens(np.asarray(tok[img_idx[order]]), np.asarray(fg[img_idx[order]]), pa["pool"])
  inv = np.empty_like(order)
  inv[order] = np.arange(N)
  tok_i, fg_i = tok_i[inv], fg_i[inv].astype(np.float32)
  refs, _ = pool_tokens(np.load(os.path.join(a.cache, "refs_tokens.npy")), np.load(os.path.join(a.cache, "refs_fg.npy")), pa["pool"])

  model = FrozenPoseModel(dim=pa["dim"], depth=pa["depth"], heads=pa["dim"] // 64, out_dim=refs.shape[-1])
  init = model.init(jax.random.PRNGKey(0), jnp.zeros((1,) + refs.shape[1:]), jnp.zeros((1, 9)))
  with open(os.path.join(a.probe, "params.msgpack"), "rb") as f:
    params = flax.serialization.from_bytes(init, f.read())
  refs_j = jnp.asarray(refs, jnp.float32)

  @jax.jit
  def energy(shape, g, tgt, w):
    pred = model.apply(params, refs_j[shape], g)
    return weighted_energy(pred, tgt.astype(jnp.float32), FG_FLOOR + w)

  g_all = pose_feat(R, t)
  img_of = np.repeat(np.arange(N), K)
  E = np.empty(len(idx), np.float32)
  for i in range(0, len(idx), a.chunk):
    sl = slice(i, i + a.chunk)
    n = len(idx[sl])
    pad = a.chunk - n
    shp = np.pad(lab[sl], (0, pad))
    im = np.pad(img_of[sl], (0, pad))
    E[sl] = np.asarray(energy(shp, np.pad(g_all[sl], ((0, pad), (0, 0))), tok_i[im], fg_i[im]))[:n]

  E, re, te, L = E.reshape(N, K), rot_err.reshape(N, K), tran_err.reshape(N, K), lab.reshape(N, K)[:, 0]
  pick_e = E.argmin(1)
  rows = np.arange(N)
  picks = {"random (baseline)": re[:, 0], "min energy": re[rows, pick_e], "oracle best": re.min(1)}
  tpicks = {"random (baseline)": te[:, 0], "min energy": te[rows, pick_e], "oracle best": te[rows, re.argmin(1)]}
  print(f"K={K} samples per image, {N} images")
  print(f"{'shape':6s} {'pick':18s} {'R mean':>7s} {'R med':>6s} {'>10deg%':>8s} {'>5deg%':>7s} {'t mean':>7s}")
  for si, name in enumerate(SHAPES + ["all"]):
    m = L == si if name != "all" else np.ones(N, bool)
    for k in picks:
      r = picks[k][m]
      print(f"{name:6s} {k:18s} {r.mean():7.3f} {np.median(r):6.3f} {100 * (r > 10).mean():8.2f} {100 * (r > 5).mean():7.2f} {tpicks[k][m].mean():7.4f}")
  # does the energy separate gross failures from good samples at all?
  bad, good = re > 10, re < 2
  if bad.any():
    rank = (E.argsort(1).argsort(1) / (K - 1))
    print(f"\nmean energy rank (0=lowest) of samples >10deg: {rank[bad].mean():.3f}; of samples <2deg: {rank[good].mean():.3f}")
    print(f"images with >=1 bad sample: {100 * bad.any(1).mean():.2f}%   images whose samples are ALL bad: {100 * bad.all(1).mean():.2f}%")
  np.save(a.samples.replace(".npz", "_energy.npy"), E)


if __name__ == "__main__":
  main()
