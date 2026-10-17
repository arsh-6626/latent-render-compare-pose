"""Numpy-only scoring shared by the oracle feature scripts (imported from both the JAX and torch envs)."""

import numpy as np

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]


def min_geodesic_deg(R, R_set):
  """(K,3,3) vs (M,3,3) -> (K,) min rotation angle in degrees."""
  tr = np.einsum("kij,mij->km", R, R_set)
  return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min(axis=1)


def score(dist, bank):
  """dist: (S, Q, N) feature distance from each query to each bank image of the same shape."""
  R_bank, R_eq, n_eq = bank["R_bank"], bank["R_eq"], bank["n_eq"]
  S, Q, N = dist.shape
  k = max(1, N // 100)
  out = {}
  for s in range(S):
    top1, prec, chance, floor = [], [], [], []
    for q in range(Q):
      d_eq = min_geodesic_deg(R_bank, R_eq[s, q, : n_eq[s, q]])  # every bank rotation's distance to the eq set
      order = np.argsort(dist[s, q])
      top1.append(d_eq[order[0]])
      prec.append((d_eq[order[:k]] < 30).mean())
      chance.append((d_eq < 30).mean())
      floor.append(d_eq.min())
    out[SHAPES[s]] = {
      "top1_err_deg_median": float(np.median(top1)),
      "top1pct_within30": float(np.mean(prec)),
      "chance_within30": float(np.mean(chance)),
      "lift": float(np.mean(prec) / max(np.mean(chance), 1e-9)),
      "grid_floor_deg_median": float(np.median(floor)),
    }
  out["mean"] = {key: float(np.mean([out[n][key] for n in SHAPES])) for key in out[SHAPES[0]]}
  return out


def print_table(name, res):
  print(f"\n== {name} ==")
  print(f"{'shape':6s} {'top1err':>8s} {'prec30':>7s} {'chance':>7s} {'lift':>6s} {'floor':>6s}")
  for k, v in res.items():
    print(f"{k:6s} {v['top1_err_deg_median']:8.1f} {v['top1pct_within30']:7.3f} {v['chance_within30']:7.3f} "
          f"{v['lift']:6.2f} {v['grid_floor_deg_median']:6.1f}")
