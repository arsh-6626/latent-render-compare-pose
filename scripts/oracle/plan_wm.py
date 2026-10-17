"""Pose estimation by planning in the world model alone: no denoiser, no pose regressor.

For each ceiling observation (500 test-5k images, cropped), search SE(3) for the pose whose WM-predicted tokens
match the observation's: CEM from uniformly random rotations, translation initialised on the ray through the crop
centre (detector-style, no GT) at a random depth. Each particle gets its own reference, retrieved at the particle's
own pose (RefBank), so nothing depends on the GT.

Diagnostic: E at the GT pose (reference retrieved at GT). If the found pose has lower energy than GT but a large
error, the energy is wrong (model failure); if higher, the search missed the GT basin (search failure).

  PYTHONPATH=src python scripts/oracle/plan_wm.py --wm $LOGS/probe/v7c_crop_geo_roll --cache $DATADIR/crop_cache \
      --ceiling $DATADIR/oracle/ceiling.npz --equiv $DATADIR/dino_cache/test-5k_equivalents.pkl --out $DATADIR/oracle/plan_v7c
"""

import argparse
import json
import os
import pickle
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from liepose.data.symsolt.dataset import SHAPE_NAMES
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, CachedSplit, weighted_energy
from liepose.exp.jepa_probe.probe_ret import CAM_C, CAM_F, CAM_Z, RefBank, RetPoseModel, init_model, wxyz_to_mat


def sym_err(R, R_eq):
  """R (n,3,3), R_eq (m,3,3) GT equivalents -> (n,) deg"""
  tr = np.einsum("nij,mij->nm", R, R_eq)
  return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min(1)


def perturb(rng, R, t, deg, tstd):
  ax = rng.normal(size=(len(R), 3))
  ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
  ang = np.radians(np.abs(rng.normal(0, deg, size=(len(R), 1))))
  return Rotation.from_rotvec(ax * ang).as_matrix() @ R, np.clip(t + rng.normal(0, tstd, size=t.shape), -1, 1)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--wm", required=True)
  p.add_argument("--cache", required=True)
  p.add_argument("--ceiling", required=True)
  p.add_argument("--equiv", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--n_init", type=int, default=2048)
  p.add_argument("--n_elite", type=int, default=32)
  p.add_argument("--n_child", type=int, default=16)
  p.add_argument("--sched_deg", type=float, nargs="+", default=[20, 10, 5, 2.5, 1.2, 0.6, 0.3])
  p.add_argument("--sched_t", type=float, nargs="+", default=[0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025])
  p.add_argument("--chunk", type=int, default=1024)
  p.add_argument("--n_obs", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  wa = SimpleNamespace(**json.load(open(os.path.join(a.wm, "args.json"))))
  assert wa.crop, "planning eval expects a crop model"
  train = CachedSplit(a.cache, wa.train_split, wa.pool)
  bank = RefBank(train, wa.lam, not wa.no_sym)
  tcrop = np.load(os.path.join(a.cache, f"{wa.train_split}_p{wa.pool}_meta.npz"))["crop"].astype(np.float32)
  model = RetPoseModel(dim=wa.dim, depth=wa.depth, heads=wa.dim // 64, out_dim=train.tokens.shape[-1], geo=wa.geo,
                       resid=getattr(wa, "resid", False))
  params = init_model(model, train.tokens.shape[1], train.tokens.shape[-1], init=a.wm)

  C = np.load(a.ceiling)
  o = np.load(os.path.join(a.cache, "ceiling_obs.npz"))
  idx, label = C["index"], C["label"]
  assert (o["index"] == idx).all()
  z_obs, w_obs, k_obs = o["tokens"].astype(np.float32), o["fg"].astype(np.float32), o["crop"].astype(np.float32)
  R_gt, t_gt = C["R"][:, 0], C["t"][:, 0]
  eqs = pickle.load(open(a.equiv, "rb"))
  n_obs = a.n_obs or len(idx)

  @jax.jit
  def energy(zr, Rr, tr, kr, Rc, tc, kg, tgt, w):
    n = Rc.shape[0]
    pred = model.apply(params, zr.astype(jnp.float32), Rr, tr, Rc, tc, kr, jnp.broadcast_to(kg, (n, 3)))
    return weighted_energy(pred, tgt[None], (FG_FLOOR + w)[None])

  def E(i, R, t):
    img, Rr, tr = bank.query(np.full(len(R), label[i]), R, t, np.full(len(R), -1))
    out = []
    for s in range(0, len(R), a.chunk):
      sl = slice(s, s + a.chunk)
      m = len(R[sl])
      pad = lambda x: np.concatenate([x, np.repeat(x[-1:], a.chunk - m, 0)]) if m < a.chunk else x  # one compile
      e = energy(pad(train.tokens[img[sl]]), pad(Rr[sl]).astype(np.float32), pad(tr[sl]).astype(np.float32),
                 pad(tcrop[img[sl]]), pad(R[sl]).astype(np.float32), pad(t[sl]).astype(np.float32), k_obs[i], z_obs[i],
                 w_obs[i])
      out.append(np.asarray(e)[:m])
    return np.concatenate(out)

  rng = np.random.default_rng(a.seed)
  res = {k: [] for k in ["rot_err", "tran_err", "E_found", "E_gt", "rot_err_iter", "top_mode_err"]}
  t0 = time.time()
  for i in range(n_obs):
    R_eq = wxyz_to_mat(eqs[idx[i]])
    s, x0, y0 = k_obs[i]
    c = np.array([x0, y0]) + 112.0 / s  # crop centre, image pixels (detector-style)
    tz = rng.uniform(-1, 1, size=(a.n_init, 1))
    txy = (c - CAM_C) / CAM_F * (tz + CAM_Z)
    R = Rotation.random(a.n_init, random_state=rng.integers(1 << 31)).as_matrix()
    t = np.clip(np.concatenate([txy, tz], 1), -1, 1)
    e = E(i, R, t)
    iters = []
    for deg, ts in zip(a.sched_deg, a.sched_t):
      el = np.argsort(e)[: a.n_elite]
      Re, te, ee = R[el], t[el], e[el]
      iters.append(float(sym_err(Re[:1], R_eq)[0]))
      Rn, tn = perturb(rng, np.repeat(Re, a.n_child, 0), np.repeat(te, a.n_child, 0), deg, ts)
      en = E(i, Rn, tn)
      R, t, e = np.concatenate([Re, Rn]), np.concatenate([te, tn]), np.concatenate([ee, en])
    b = np.argmin(e)
    el = np.argsort(e)[: a.n_elite]
    res["rot_err"].append(float(sym_err(R[b : b + 1], R_eq)[0]))
    res["tran_err"].append(float(np.linalg.norm(t[b] - t_gt[i])))
    res["E_found"].append(float(e[b]))
    res["E_gt"].append(float(E(i, R_gt[i : i + 1], t_gt[i : i + 1])[0]))
    res["rot_err_iter"].append(iters + [res["rot_err"][-1]])
    res["top_mode_err"].append(float(np.median(sym_err(R[el], R_eq))))
    if (i + 1) % 25 == 0:
      r = np.array(res["rot_err"])
      print(f"{i + 1}/{n_obs}  rot mean {r.mean():.2f} med {np.median(r):.2f} <2deg {(r < 2).mean():.2f} "
            f">15deg {(r > 15).mean():.2f}  {time.time() - t0:.0f}s", flush=True)

  r, tt = np.array(res["rot_err"]), np.array(res["tran_err"])
  ef, eg = np.array(res["E_found"]), np.array(res["E_gt"])
  lab = label[:n_obs]
  bad = r > 15
  summ = {"wm": a.wm, "n_obs": n_obs, "rot_mean": float(r.mean()), "rot_median": float(np.median(r)),
          "rot_lt1": float((r < 1).mean()), "rot_lt2": float((r < 2).mean()), "rot_lt5": float((r < 5).mean()),
          "rot_gt15": float(bad.mean()), "tran_mean": float(tt.mean()),
          "rot_mean_excl_gt15": float(r[~bad].mean()) if (~bad).any() else None,
          "per_shape_mean": {SHAPE_NAMES[s]: float(r[lab == s].mean()) for s in np.unique(lab)},
          "per_shape_median": {SHAPE_NAMES[s]: float(np.median(r[lab == s])) for s in np.unique(lab)},
          "gt15: model failure (E_found<E_gt)": float((ef[bad] < eg[bad]).mean()) if bad.any() else None,
          "gt15: search failure (E_found>=E_gt)": float((ef[bad] >= eg[bad]).mean()) if bad.any() else None,
          "all: E_found<E_gt": float((ef < eg).mean()),
          "rot_err_by_iter_mean": np.array(res["rot_err_iter"]).mean(0).tolist(),
          "sec": time.time() - t0}
  print(json.dumps(summ, indent=2), flush=True)
  json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
  np.savez(os.path.join(a.out, "per_obs.npz"), **{k: np.array(v) for k, v in res.items()}, label=lab, index=idx[:n_obs])


if __name__ == "__main__":
  main()
