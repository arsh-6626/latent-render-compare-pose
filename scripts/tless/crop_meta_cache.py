"""Per-row crop meta for the T-LESS token cache, from export_crops.py --meta_only --geometry shards (same seed and
loader as the cache's export: rows must match the cache's index and translation_scaled exactly).

Writes into the cache dir, per split:
  {split}_crop_meta.npy  (N, 10) tless_field.crop_meta10: the crop camera
  {split}_ts_true.npy    (N, 3) translation_scaled of the true 3D centre (translation_allo). LiePose's
                         translation_scaled encodes the offset of the annotated 2D centroid, which differs from the
                         projection of the 3D centre on ~1/4 of the training crops (depth unaffected); the field is
                         fitted at the true pose, the diffusion keeps LiePose's parametrisation.

  PYTHONPATH=src:lib/bop_toolkit python scripts/tless/crop_meta_cache.py --crops $D/tless_crops_meta --cache $D/tless_tokens
"""

import argparse
import glob
import os

import numpy as np

from liepose.exp.jepa_probe.tless_field import TRAN_OFFSET, TRAN_SCALE, crop_meta10


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--crops", required=True)
  p.add_argument("--cache", required=True)
  p.add_argument("--splits", nargs="+", default=["train", "test"])
  a = p.parse_args()
  for split in a.splits:
    sh = [np.load(f) for f in sorted(glob.glob(os.path.join(a.crops, split, "shard_*.npz")))]
    d = {k: np.concatenate([s[k] for s in sh]) for k in sh[0].files}
    m = np.load(os.path.join(a.cache, f"{split}_meta.npz"))
    n = len(m["index"])
    assert len(d["index"]) >= n, (split, len(d["index"]), n)
    d = {k: v[:n] for k, v in d.items()}
    assert (d["index"] == m["index"]).all(), f"{split}: index misaligned with the token cache"
    dts = np.abs(d["translation_scaled"] - m["translation_scaled"]).max()
    assert dts < 1e-5, f"{split}: translation_scaled misaligned ({dts})"
    k = crop_meta10(d)
    np.save(os.path.join(a.cache, f"{split}_crop_meta.npy"), k)
    if "translation_allo" not in d:  # test: the crop camera only (no 3D ground truth at inference)
      print(f"{split}: {n} rows aligned; crop meta only", flush=True)
      continue
    tt, K = d["translation_allo"].astype(np.float64), d["cam"].astype(np.float64)
    u = np.einsum("nij,nj->ni", K, tt)
    u = u[:, :2] / u[:, 2:]
    ctr = k[:, 1:3] + 112.0 / k[:, :1]
    sc, off = np.asarray(TRAN_SCALE), np.asarray(TRAN_OFFSET)
    ts = d["translation_scaled"].astype(np.float64).copy()
    ts[:, :2] = (u - ctr) / k[:, 3:5] / sc[:2] - off[:2]
    moved = np.abs(ts - d["translation_scaled"]).max(1) > 1e-4
    np.save(os.path.join(a.cache, f"{split}_ts_true.npy"), ts.astype(np.float32))
    print(f"{split}: {n} rows aligned; true-centre translation differs from LiePose's on {moved.mean():.3f}", flush=True)


if __name__ == "__main__":
  main()
