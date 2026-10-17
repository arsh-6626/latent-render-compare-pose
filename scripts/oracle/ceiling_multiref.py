"""Refinement ceiling with K references per observation (probe_ret / v7 models): is the predictor's error
reference-specific noise that averages away?

The K nearest distinct training images to the same noisy-GT query each predict every candidate; we score
  mean_E: the mean of the K energies, and
  mean_z: the energy of the mean prediction,
and report the ceiling pick error for K = 1, 2, 4, 8 (K = 1 reproduces the in-training eval).

  PYTHONPATH=src python scripts/oracle/ceiling_multiref.py --wm $LOGS/probe/v7_ret_geo_roll \
      --cache $DATADIR/dino_cache --ceiling $DATADIR/oracle/ceiling.npz --out $DATADIR/oracle/multiref_v7.json
"""

import argparse
import json
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, CachedSplit, pool_tokens, weighted_energy
from liepose.exp.jepa_probe.probe_ret import RefBank, RetPoseModel, init_model, noisy_pose


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--wm", required=True)
  p.add_argument("--cache", required=True)
  p.add_argument("--ceiling", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
  a = p.parse_args()
  wa = SimpleNamespace(**json.load(open(os.path.join(a.wm, "args.json"))))
  train = CachedSplit(a.cache, wa.train_split, wa.pool)
  bank = RefBank(train, wa.lam, not wa.no_sym)
  model = RetPoseModel(dim=wa.dim, depth=wa.depth, heads=wa.dim // 64, out_dim=train.tokens.shape[-1], geo=wa.geo,
                       **({"resid": True} if getattr(wa, "resid", False) else {}))
  params = init_model(model, train.tokens.shape[1], train.tokens.shape[-1], init=a.wm)

  C = np.load(a.ceiling)
  idx, label = C["index"], C["label"]
  rot_err, kind = C["rot_err"], C["kind"]
  crop = getattr(wa, "crop", False)
  if crop:  # zoomed crops (crop_encode.py): observations and references carry their crop (s, x0, y0)
    o = np.load(os.path.join(a.cache, "ceiling_obs.npz"))
    z, w, kg = o["tokens"], o["fg"].astype(np.float32), o["crop"]
    train_crop = np.load(os.path.join(a.cache, f"{wa.train_split}_p{wa.pool}_meta.npz"))["crop"]
  else:
    tok = np.load(os.path.join(a.cache, "test-5k_tokens.npy"), mmap_mode="r")
    fg = np.load(os.path.join(a.cache, "test-5k_fg.npy"), mmap_mode="r")
    z, w = pool_tokens(np.asarray(tok[idx]), np.asarray(fg[idx]), wa.pool)
  Rq, tq = noisy_pose(np.random.default_rng(123), C["R"][:, 0], C["t"][:, 0], wa.ref_noise_deg, wa.ref_noise_t)
  K = max(a.ks)
  img, Rr, tr = bank.query_k(label, Rq, tq, K)
  Rc, tc = jnp.asarray(C["R"], jnp.float32), jnp.asarray(C["t"], jnp.float32)

  @jax.jit
  def predict(zr, Rr, tr, Rc, tc, kr=None, kg=None):
    """K refs x N candidates -> (K, N, P, D)"""
    k, n = zr.shape[0], Rc.shape[0]
    rep = lambda x: jnp.repeat(x, n, axis=0)
    if kr is not None:
      kr, kg = rep(kr), jnp.broadcast_to(kg, (k * n, 3))
    pred = model.apply(params, rep(zr.astype(jnp.float32)), rep(Rr), rep(tr), jnp.tile(Rc, (k, 1, 1)), jnp.tile(tc, (k, 1)),
                       kr, kg)
    return pred.reshape(k, n, *pred.shape[1:])

  E = {f"{m}_K{k}": [] for m in ["mean_E", "mean_z"] for k in a.ks}
  for i in range(len(idx)):
    ks = (jnp.asarray(train_crop[img[i]]), jnp.asarray(kg[i])) if crop else ()
    pr = predict(jnp.asarray(train.tokens[img[i]]), jnp.asarray(Rr[i], jnp.float32), jnp.asarray(tr[i], jnp.float32), Rc[i], tc[i],
                 *ks)
    tgt, ww = jnp.asarray(z[i], jnp.float32), FG_FLOOR + jnp.asarray(w[i], jnp.float32)
    e = weighted_energy(pr, tgt[None, None], ww[None, None])  # (K, N)
    for k in a.ks:
      E[f"mean_E_K{k}"].append(np.asarray(e[:k].mean(0)))
      E[f"mean_z_K{k}"].append(np.asarray(weighted_energy(pr[:k].mean(0), tgt[None], ww[None])))
  rot = kind == "rot"
  res = {"wm": a.wm}
  for name, v in E.items():
    v = np.stack(v)
    sel = np.array([rot_err[i, np.nonzero(rot[i])[0][np.argmin(v[i, rot[i]])]] for i in range(len(v))])
    m = rot & (rot_err >= 0.75) & (rot_err < 1.5)
    res[name] = {"pick_rot_err": float(sel.mean()),
                 "P(E_pert>E_gt) 0.75-1.5": float((v[m] > np.broadcast_to(v[:, :1], v.shape)[m]).mean())}
    print(f"{name:12s} pick {res[name]['pick_rot_err']:.3f}  P@0.75-1.5 {res[name]['P(E_pert>E_gt) 0.75-1.5']:.3f}", flush=True)
  json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
  main()
