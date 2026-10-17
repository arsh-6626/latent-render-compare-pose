"""Export SPEED+ crops (same crop, augmentation and pose parametrisation as LiePose training) to uint8 shards,
for DINOv2 encoding, world-model training and our sampling/evaluation scripts.

Each shard: image (N,224,224,3) uint8, object_id, rotation_allo / rotation (wxyz), translation_scaled,
translation (m), bbox (xyxy), bbox_center, dzi_size, focal_ratio, index (row in the split json).
No `mask` (SPEED+ has none; encode_crops.py then uses uniform foreground weights).

  DATADIR=... PYTHONPATH=src python scripts/speedplus/export_crops.py --split train --copies 2 --out $DATADIR/speedplus_crops/train
  DATADIR=... PYTHONPATH=src python scripts/speedplus/export_crops.py --split lightbox --out $DATADIR/speedplus_crops/lightbox
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from liepose.data.bop.dataset import collate_fn
from liepose.data.speedplus.dataset import SpeedPlusPoseDataset
from liepose.data.utils import seed_worker
from liepose.exp.speedplus.run import AUGMENT, TRAN_OFFSET, TRAN_SCALE

KEYS = ["index", "object_id", "rotation_allo", "rotation", "translation_scaled", "translation", "bbox", "bbox_center",
        "dzi_size", "focal_ratio"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--split", choices=["train", "val", "lightbox", "sunlamp"], required=True)
  p.add_argument("--copies", type=int, default=1, help="augmented crops per training image (train split only)")
  p.add_argument("--boxes", default=None, help=".npy (N,4) xyxy detector boxes; default: GT keypoint boxes")
  p.add_argument("--shard", type=int, default=8192)
  p.add_argument("--batch", type=int, default=64)
  p.add_argument("--workers", type=int, default=12)
  p.add_argument("--seed", type=int, default=123)
  p.add_argument("--out", required=True)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  is_train = a.split == "train"
  boxes = np.load(a.boxes) if a.boxes else None
  ds = SpeedPlusPoseDataset(a.split, is_train=is_train, augment=AUGMENT, tran_scale=TRAN_SCALE,
                            tran_offset=TRAN_OFFSET, boxes=boxes)
  order = np.arange(len(ds))
  if is_train:
    rng = np.random.default_rng(a.seed)
    order = np.concatenate([rng.permutation(len(ds)) for _ in range(a.copies)])
  g = torch.Generator()
  g.manual_seed(a.seed)
  dl = DataLoader(Subset(ds, order.tolist()), batch_size=a.batch, shuffle=False, num_workers=a.workers,
                  worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn)

  buf, shard_id, done = {k: [] for k in ["image"] + KEYS}, 0, 0

  def flush():
    nonlocal buf, shard_id
    if not buf["image"]:
      return
    np.savez(os.path.join(a.out, f"shard_{shard_id:04d}.npz"), **{k: np.concatenate(v) for k, v in buf.items()})
    shard_id += 1
    buf = {k: [] for k in buf}

  for batch in tqdm(dl, desc=f"export {a.split}"):
    buf["image"].append(np.clip(np.asarray(batch["image"]), 0, 255).astype(np.uint8))
    for k in KEYS:
      buf[k].append(np.asarray(batch[k]))
    done += len(batch["image"])
    if sum(len(x) for x in buf["image"]) >= a.shard:
      flush()
  flush()
  print(f"exported {done} crops in {shard_id} shards to {a.out}")


if __name__ == "__main__":
  main()
