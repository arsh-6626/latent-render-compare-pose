"""Official ESA of N-sample consensus on a SPEED+ split, for any model's saved samples (test_liepose.py --n_slices,
test_consensus.py): <split>_samples.npz with index (n,), R (n, S, 3, 3) egocentric, t (n, S, 3) camera frame (m).
Consensus (GT-free; select_wm2.py's rule): the sample with the most other samples within 10 deg (ties: the smaller
mean geodesic distance). Scored with esa_score.esa against the crop shards' GT (scorer only): sample 0 (= one
sample) and the consensus pick; also the share of rotation errors > 10 / > 90 deg.

  python scripts/speedplus/consensus_score.py --samples <dir>/lightbox_samples.npz --crops $DATADIR/speedplus_crops/lightbox --out <json>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esa_score import esa  # noqa: E402


def consensus(R):
  """R (n, S, 3, 3) -> (n,) index of the consensus sample"""
  G = np.degrees(np.arccos(np.clip((np.einsum("nsij,ntij->nst", R, R) - 1) / 2, -1, 1)))
  return ((G < 10).sum(-1) - 1e-3 * G.mean(-1)).argmax(1)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--samples", required=True)
  p.add_argument("--crops", required=True)
  p.add_argument("--out", default=None)
  p.add_argument("--allow_partial", action="store_true", help="val sweeps on the first n crops only (never for test splits)")
  a = p.parse_args()
  sm = np.load(a.samples)
  idx, R, t = sm["index"].astype(np.int64), sm["R"].astype(np.float64), sm["t"].astype(np.float64)
  sh = [np.load(f) for f in sorted(glob.glob(os.path.join(a.crops, "shard_*.npz")))]
  ci = np.concatenate([x["index"] for x in sh])
  pos = {int(i): r for r, i in enumerate(ci)}
  g = np.array([pos[int(i)] for i in idx])
  assert a.allow_partial or len(np.unique(idx)) == len(ci), "samples do not cover the full split"
  assert not (a.allow_partial and os.path.basename(os.path.normpath(a.crops)) in ("lightbox", "sunlamp")), "test splits are scored in full"
  R_gt = Rotation.from_quat(np.roll(np.concatenate([x["rotation"] for x in sh])[g], -1, -1)).as_matrix()
  t_gt = np.concatenate([x["translation"] for x in sh])[g].astype(np.float64)
  n, S = R.shape[:2]
  ar = np.arange(n)
  res = {"samples": a.samples, "n": int(n), "S": int(S)}
  for name, c in [("sample0", np.zeros(n, int)), ("consensus", consensus(R))]:
    r = esa(R[ar, c], t[ar, c], R_gt, t_gt)
    e = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", R[ar, c], R_gt) - 1) / 2, -1, 1)))
    r.update(e_q_gt10=float((e > 10).mean()), e_q_gt90=float((e > 90).mean()))
    res[name] = r
    print(name, json.dumps(r), flush=True)
  if a.out:
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
