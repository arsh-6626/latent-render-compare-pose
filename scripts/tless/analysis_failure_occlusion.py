"""T-LESS analyses on saved LiePose samples (no training):
(1) failure detection: how well each confidence signal flags bad poses (AUROC, risk-coverage);
(2) occlusion: pick accuracy per visibility bucket.

Signals per sample: world-model energy (low = confident), symmetry-aware agreement with the other samples
(mean angle to the rest; low = confident), joint-scorer score; per crop: set dispersion.

  CACHE_ROOT_PATH=... DATADIR=... PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/tless/analysis_failure_occlusion.py \
      --samples tless_verifier_400k_noema_samples.npz --scorer_dir $LOGS/tless_scorer_v1 --out analysis.json
"""

import argparse
import json
import os

import flax
import jax
import jax.numpy as jnp
import numpy as np

import liepose.exp.bop.dataset_registry  # noqa: F401
from liepose.data.bop.dataset import TestDataset
from liepose.exp.bop.run import build_explorations

import sys
sys.path.insert(0, os.path.dirname(__file__))
from scorer import Scorer  # noqa: E402

AGREE_COL = 66  # feature column: symmetry-aware mean angle to the other samples / 30


def auroc(score_bad, is_bad):
  """P(score of a bad sample > score of a good sample); ties count half."""
  s = np.asarray(score_bad, np.float64)
  y = np.asarray(is_bad, bool)
  if y.all() or (~y).all():
    return float("nan")
  order = np.argsort(s, kind="mergesort")
  ranks = np.empty(len(s))
  ranks[order] = np.arange(1, len(s) + 1)
  # average ranks for ties
  _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
  sums = np.bincount(inv, weights=ranks)
  ranks = (sums / cnt)[inv]
  n1, n0 = y.sum(), (~y).sum()
  return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def risk_at_coverage(conf_bad, err, covs=(1.0, 0.9, 0.8, 0.7, 0.5)):
  """Keep the most-confident fraction (lowest conf_bad); report mean error and failure rate kept."""
  order = np.argsort(conf_bad)
  out = {}
  for c in covs:
    keep = order[: max(1, int(round(c * len(order))))]
    out[f"{int(c * 100)}%"] = {"rot_mean": float(err[keep].mean()), "gt10": float(100 * (err[keep] > 10).mean()),
                                "gt30": float(100 * (err[keep] > 30).mean())}
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--samples", required=True)
  p.add_argument("--scorer_dir", required=True)
  p.add_argument("--out", required=True)
  a = p.parse_args()

  S = dict(np.load(a.samples))
  F = np.load(os.path.join(a.scorer_dir, "features.npz"))["test"]
  re, te, E = S["rot_err"], S["tran_err"], S["energy"]
  N, K = re.shape
  assert F.shape[:2] == (N, K)
  model = Scorer()
  init = model.init(jax.random.PRNGKey(0), jnp.zeros((1,) + F.shape[1:]))
  params = flax.serialization.from_bytes(init, open(os.path.join(a.scorer_dir, "scorer.msgpack"), "rb").read())
  sc = np.concatenate([np.asarray(model.apply(params, jnp.asarray(F[i : i + 1024]))) for i in range(0, N, 1024)])
  agree = F[:, :, AGREE_COL]  # higher = disagrees more

  # visibility per crop
  e = build_explorations()
  ds = TestDataset(e.test_datasets[0], batch_size=8, include_ids=None, input_res=224, is_train=False,
                   tran_scale=e.tran_scale[0], tran_offset=e.tran_offset[0], use_reproj=False, with_coord=False)
  dd = ds.dataset.dataset.dataset_dicts
  vis = np.array([dd[int(i)]["inst_infos"]["visib_fract"] for i in S["index"]])

  res = {"n_crops": int(N), "k": int(K)}
  # ---- (1) per-sample failure detection (all N*K samples) ----
  flat = lambda x: x.reshape(-1)
  # within-crop relative versions (remove per-crop offset)
  E_rel = E - E.min(1, keepdims=True)
  sc_rel = -(sc - sc.max(1, keepdims=True))
  sig = {"energy": E, "energy_rel_to_set_min": E_rel, "disagreement_voting": agree, "neg_scorer": -sc,
         "neg_scorer_rel": sc_rel}
  res["sample_auroc"] = {}
  for thr in (10, 30):
    res["sample_auroc"][f">{thr}deg"] = {k: auroc(flat(v), flat(re) > thr) for k, v in sig.items()}

  # ---- per-crop confidence of the FINAL pick (risk-coverage) ----
  rows = np.arange(N)
  picks = {"random(LiePose)": np.zeros(N, int), "voting": agree.argmin(1), "min_energy": E.argmin(1), "scorer": sc.argmax(1)}
  disp = agree.mean(1)  # set dispersion: average symmetry-aware angle between samples
  res["crop_confidence"] = {}
  for pk, j in picks.items():
    err = re[rows, j]
    confs = {"energy_of_pick": E[rows, j], "disagreement_of_pick": agree[rows, j], "set_dispersion": disp,
             "neg_scorer_of_pick": -sc[rows, j]}
    res["crop_confidence"][pk] = {
      "auroc_gt10": {k: auroc(v, err > 10) for k, v in confs.items()},
      "auroc_gt30": {k: auroc(v, err > 30) for k, v in confs.items()},
      "risk_coverage": {k: risk_at_coverage(v, err) for k, v in confs.items()},
    }

  # ---- (2) occlusion buckets ----
  oracle = np.argmin(re / 5.0 + te / 0.1, 1)
  picks["oracle"] = oracle
  buckets = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]
  res["visibility"] = {}
  for lo, hi in buckets:
    m = (vis >= lo) & (vis < hi)
    row = {"n": int(m.sum())}
    for pk, j in picks.items():
      r = re[rows, j][m]
      row[pk] = {"rot_mean": float(r.mean()), "rot_median": float(np.median(r)), "gt10": float(100 * (r > 10).mean()),
                 "gt30": float(100 * (r > 30).mean())}
    res["visibility"][f"{lo:.1f}-{min(hi, 1.0):.1f}"] = row
  json.dump(res, open(a.out, "w"), indent=2)

  # ---- print ----
  print("\n(1a) per-sample failure detection AUROC (higher = better at flagging bad samples)")
  for thr, d in res["sample_auroc"].items():
    print(f"  {thr:7s} " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()))
  print("\n(1b) confidence of the final pick: AUROC for '>30deg' and error after keeping the most-confident 80%")
  for pk, d in res["crop_confidence"].items():
    print(f"  pick={pk}")
    for k in d["auroc_gt30"]:
      rc = d["risk_coverage"][k]
      print(f"    {k:22s} AUROC>30 {d['auroc_gt30'][k]:.3f}  >10 {d['auroc_gt10'][k]:.3f} | all: R {rc['100%']['rot_mean']:.2f} "
            f">30 {rc['100%']['gt30']:.2f}% | top80%: R {rc['80%']['rot_mean']:.2f} >30 {rc['80%']['gt30']:.2f}% | "
            f"top50%: R {rc['50%']['rot_mean']:.2f} >30 {rc['50%']['gt30']:.2f}%")
  print("\n(2) by visibility (visible fraction of the object): rot mean / >30deg %")
  for b, row in res["visibility"].items():
    print(f"  vis {b} (n={row['n']}): " + "  ".join(f"{pk} {row[pk]['rot_mean']:.1f}/{row[pk]['gt30']:.1f}%" for pk in picks))


if __name__ == "__main__":
  main()
