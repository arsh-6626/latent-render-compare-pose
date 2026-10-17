"""Pick one of K samples per image from a WM-native denoiser run (cmp.run with eval_samples=K, eval_tag=_k16).

Rules: random (= the denoiser alone), translation medoid (voting), min self-energy (the energy denoiser's own
learned E at the last step), min v4 world-model energy (if verifier_eval.py was run on the same file),
and the oracle best sample. Also: how well each score flags images whose random pick is a gross failure.

  python scripts/wmden_select.py <eval_dir>/errors.npz
  (optional, first) PYTHONPATH=src python scripts/verifier_eval.py --samples <eval_dir>/errors.npz \
      --cache $DATADIR/dino_cache --probe $LOGS/probe/v4_trans      # writes errors_energy.npy
"""

import argparse
import json
import os

import numpy as np

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]


def auroc(score, y):
  """P(score of a positive > score of a negative), ties count half."""
  pos, neg = score[y], score[~y]
  if len(pos) == 0 or len(neg) == 0:
    return float("nan")
  order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
  ranks = np.empty(len(order))
  ranks[order] = np.arange(1, len(order) + 1)
  return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("errors")
  a = p.parse_args()
  d = np.load(a.errors)
  K = int(d["n_slices"])
  N = len(d["index"]) // K
  idx = d["index"].reshape(N, K)
  assert (idx == idx[:, :1]).all(), "samples of one image must be contiguous"
  re, te, L = d["rot_err"].reshape(N, K), d["tran_err"].reshape(N, K), d["label"].reshape(N, K)[:, 0]
  t = d["t"].reshape(N, K, 3)
  dt = np.linalg.norm(t[:, :, None] - t[:, None], axis=-1).sum(-1)  # (N,K) total distance to the other samples
  rows = np.arange(N)
  picks = {"random": np.zeros(N, int), "medoid_t (voting)": dt.argmin(1)}
  scores = {"voting disagreement": dt.min(1) / (K - 1)}
  if "self_energy" in d:
    Es = d["self_energy"].reshape(N, K)
    picks["min self-energy"] = Es.argmin(1)
    scores["self-energy of pick"] = Es.min(1)
  v4 = a.errors.replace(".npz", "_energy.npy")
  if os.path.isfile(v4):
    Ev = np.load(v4).reshape(N, K)
    picks["min v4 energy"] = Ev.argmin(1)
    scores["v4 energy of pick"] = Ev.min(1)
  picks["oracle"] = re.argmin(1)

  res = {"K": K, "images": N, "picks": {}, "failure_auroc(random pick >10deg)": {}}
  print(f"K={K}, {N} images")
  print(f"{'shape':6s} {'pick':20s} {'R mean':>7s} {'R med':>6s} {'>10deg%':>8s} {'t mean':>7s}")
  for si, name in enumerate(SHAPES + ["all"]):
    m = L == si if name != "all" else np.ones(N, bool)
    for k, j in picks.items():
      r, tt = re[rows, j][m], te[rows, j][m]
      row = {"rot_mean": float(r.mean()), "rot_median": float(np.median(r)), "rot_gt10_pct": float(100 * (r > 10).mean()),
             "tran_mean": float(tt.mean())}
      res["picks"].setdefault(name, {})[k] = row
      print(f"{name:6s} {k:20s} {row['rot_mean']:7.3f} {row['rot_median']:6.3f} {row['rot_gt10_pct']:8.2f} {row['tran_mean']:7.4f}")
  fail = re[:, 0] > 10
  print(f"\nfailure detection (random pick >10deg, {100 * fail.mean():.2f}% of images): AUROC")
  for k, s in scores.items():
    res["failure_auroc(random pick >10deg)"][k] = auroc(s, fail)
    print(f"  {k:24s} {res['failure_auroc(random pick >10deg)'][k]:.3f}")
  out = a.errors.replace(".npz", "_select.json")
  json.dump(res, open(out, "w"), indent=2)
  print(f"saved {out}")


if __name__ == "__main__":
  main()
