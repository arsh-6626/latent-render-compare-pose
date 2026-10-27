"""Apply LiePose's SPEED+ training colour augmentation (build_color_augment, applied with random_color_prob as in
liepose.exp.speedplus.run.AUGMENT) to exported crop shards, keeping every other field; a fixed seed. For the
appearance-sensitivity check on the synthetic validation split (no lightbox / sunlamp image is read).

  DATADIR=... PYTHONPATH=src python scripts/speedplus/augment_crops.py --crops $DATADIR/speedplus_crops/val \
      --out $DATADIR/speedplus_crops/val_aug
"""

import argparse
import glob
import os

import imgaug
import numpy as np

from liepose.data.bop.data_utils.augment import build_color_augment
from liepose.exp.speedplus.run import AUGMENT


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--crops", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--prob", type=float, default=AUGMENT["random_color_prob"])
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  imgaug.seed(a.seed)
  rng = np.random.default_rng(a.seed)
  aug = build_color_augment()
  n_aug = 0
  for s in sorted(glob.glob(os.path.join(a.crops, "shard_*.npz"))):
    d = dict(np.load(s))
    img = d["image"].copy()
    on = rng.random(len(img)) < a.prob
    for i in np.flatnonzero(on):
      img[i] = aug.augment_image(img[i])
    d["image"], d["color_aug"] = img, on
    n_aug += int(on.sum())
    np.savez(os.path.join(a.out, os.path.basename(s)), **d)
    print(f"{os.path.basename(s)}: {len(img)} crops, {int(on.sum())} augmented", flush=True)
  print(f"done: {n_aug} augmented (prob {a.prob})")


if __name__ == "__main__":
  main()
