"""Raw scaled-translation statistics of a BOP training set through LiePose's own loader (tran_scale = 1, tran_offset = 0):
raw = [dx / bbox_w, dy / bbox_h, z / focal_ratio]. Finding (2026-10-30): bop/run.py's T-LESS constants (from upstream LiePose) standardise the T-LESS *test* set
(translation_scaled there has mean 0 / std 1 to 3 decimals), not the training crops. New datasets take their constants
from training crops only (no augmentation, robust statistics against the heavy x / y tails).

  DATADIR=... PYTHONPATH=src:lib/bop_toolkit python scripts/tless/tran_stats.py --datasets tless-train_pbr tless-train_primesense --n 20000
"""

import argparse

import numpy as np

import liepose.exp.bop.dataset_registry  # noqa: F401
from liepose.data.bop.dataset import TrainDataset


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--datasets", nargs="+", required=True)
  p.add_argument("--n", type=int, default=20000)
  p.add_argument("--augment", type=int, default=1, help="LiePose's training augmentation (dynamic zoom-in) on / off")
  p.add_argument("--workers", type=int, default=12)
  a = p.parse_args()
  aug = dict(random_color_prob=0.8, random_bg_prob=0.5, random_bbox_dzi=bool(a.augment), dzi_pad_scale=1.5,
             dzi_scale_ratio=0.25, dzi_shift_ratio=0.25, dzi_type="uniform")
  ds = TrainDataset(a.datasets, batch_size=64, seed=0, num_workers=a.workers, is_train=True, augment=aug, equivalent_rots=False,
                    include_ids=None, input_res=224, tran_scale=[1.0, 1.0, 1.0], tran_offset=[0.0, 0.0, 0.0], use_reproj=False,
                    with_coord=False)
  xs, n = [], 0
  for b in ds:
    xs.append(np.asarray(b["translation_scaled"], np.float64))
    n += len(xs[-1])
    if n >= a.n:
      break
  x = np.concatenate(xs)[: a.n]
  m, s = x.mean(0), x.std(0)
  print("n", len(x), "augment", a.augment)
  print("tran_scale  (std)       ", np.round(s, 6).tolist())
  print("tran_offset (mean / std)", np.round(m / s, 6).tolist())
  med = np.median(x, 0)
  mad = 1.4826 * np.median(np.abs(x - med), 0)  # robust std (training crops have heavy x / y tails)
  print("robust scale  (1.4826 MAD)    ", np.round(mad, 6).tolist())
  print("robust offset (median / MAD)  ", np.round(med / mad, 6).tolist())
  print("percentiles 1/50/99 x", np.percentile(x[:, 0], [1, 50, 99]).round(4).tolist(), "y", np.percentile(x[:, 1], [1, 50, 99]).round(4).tolist(),
        "z", np.percentile(x[:, 2], [1, 50, 99]).round(4).tolist())


if __name__ == "__main__":
  main()
