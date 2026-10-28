"""Export LiePose's own T-LESS crops (same augmentation, masking and pose parametrisation as the
denoiser) to uint8 shards, for DINOv2 encoding and world-model training.

Each shard: image (N,224,224,3) uint8 already multiplied by the visible mask, mask (N,224,224) bool,
object_id, rotation_allo (wxyz), translation_scaled, index (dataset index; test split only).

  DATADIR=... PYTHONPATH=src python scripts/tless/export_crops.py --split train --n 200000 --out $DATADIR/tless_crops/train
  DATADIR=... PYTHONPATH=src python scripts/tless/export_crops.py --split test --out $DATADIR/tless_crops/test
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

import liepose.exp.bop.dataset_registry  # noqa: F401  (registers tless datasets)
from liepose.data.bop.dataset import TestDataset, TrainDataset
from liepose.exp.bop.run import build_explorations

KEYS = ["object_id", "rotation_allo", "translation_scaled"]
# --geometry: what the world model's geometry tokens need (crop intrinsics, metric translation, crop box)
GEO_KEYS = ["cam_crop", "translation_allo", "bbox_center", "bbox_size"]
# --meta: the feature-field world model's crop camera (sp_geometry.crop_k's box quantities + the image intrinsics)
META_KEYS = ["bbox", "bbox_center", "dzi_size", "focal_ratio", "cam"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--split", choices=["train", "test"], required=True)
  p.add_argument("--n", type=int, default=200000, help="train crops to export")
  p.add_argument("--shard", type=int, default=8192)
  p.add_argument("--batch", type=int, default=64)
  p.add_argument("--workers", type=int, default=12)
  p.add_argument("--out", required=True)
  p.add_argument("--geometry", action="store_true", help="also store GEO_KEYS per crop")
  p.add_argument("--meta", action="store_true", help="also store META_KEYS per crop")
  p.add_argument("--meta_only", action="store_true", help="META_KEYS (+ KEYS, index) only, no image / mask")
  p.add_argument("--seed", type=int, default=123, help="train augmentation seed")
  a = p.parse_args()
  keys = KEYS + (GEO_KEYS if a.geometry else [])
  keys += [k for k in META_KEYS if (a.meta or a.meta_only) and k not in keys]
  os.makedirs(a.out, exist_ok=True)
  e = build_explorations()
  common = dict(include_ids=None, input_res=224, tran_scale=e.tran_scale[0], tran_offset=e.tran_offset[0],
                use_reproj=False, with_coord=False)

  if a.split == "train":
    ds = TrainDataset(e.train_datasets[0], batch_size=a.batch, seed=a.seed, num_workers=a.workers, is_train=True,
                      augment=dict(random_color_prob=0.8, random_bg_prob=0.5, random_bbox_dzi=True, dzi_pad_scale=1.5,
                                   dzi_scale_ratio=0.25, dzi_shift_ratio=0.25, dzi_type="uniform"),
                      equivalent_rots=False, **common)
    it, total = iter(ds), a.n
  else:
    ds = TestDataset(e.test_datasets[0], batch_size=a.batch, is_train=False, **common)
    it, total = iter(ds), None

  buf, shard_id, done = {k: [] for k in ([] if a.meta_only else ["image", "mask"]) + ["index"] + keys}, 0, 0

  def flush():
    nonlocal buf, shard_id
    if not buf["index"]:
      return
    out = {k: np.concatenate(v) for k, v in buf.items() if v}
    np.savez(os.path.join(a.out, f"shard_{shard_id:04d}.npz"), **out)
    shard_id += 1
    buf = {k: [] for k in buf}

  pbar = tqdm(total=total, desc=f"export {a.split}")
  for batch in it:
    n_b = len(batch["index"])
    if not a.meta_only:
      img = np.asarray(batch["image"], np.float32)
      mask = np.asarray(batch["mask"], np.float32)
      buf["image"].append(np.clip(img * mask, 0, 255).astype(np.uint8))
      buf["mask"].append(mask[..., 0] > 0.5)
    buf["index"].append(np.asarray(batch["index"]))
    for k in keys:
      buf[k].append(np.asarray(batch[k]))
    done += n_b
    pbar.update(n_b)
    if sum(len(x) for x in buf["index"]) >= a.shard:
      flush()
    if total is not None and done >= total:
      break
  flush()
  print(f"exported {done} crops in {shard_id} shards to {a.out}")


if __name__ == "__main__":
  main()
