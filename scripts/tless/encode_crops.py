"""Encode exported T-LESS crop shards with frozen DINOv2 ViT-S/14 (torch-CUDA env, e.g. `jepa`).

Writes <out>/<split>_tokens.npy (N,64,384) fp16 (2x2-pooled patch tokens), <split>_fg.npy (N,64)
visible-mask weights (max-pooled, dilated by one patch) and <split>_meta.npz (object_id, rotation_allo,
translation_scaled, index).

  python scripts/tless/encode_crops.py --crops $DATADIR/tless_crops/test --split test --out $DATADIR/tless_tokens
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

META = ["object_id", "rotation_allo", "translation_scaled", "index"]


@torch.no_grad()
def main():
  p = argparse.ArgumentParser()
  p.add_argument("--crops", required=True)
  p.add_argument("--split", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--bs", type=int, default=256)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  shards = sorted(glob.glob(os.path.join(a.crops, "shard_*.npz")))
  n = sum(len(np.load(s, mmap_mode="r")["object_id"]) for s in shards)
  net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
  mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
  tok = np.lib.format.open_memmap(os.path.join(a.out, f"{a.split}_tokens.npy"), "w+", np.float16, (n, 64, 384))
  fg = np.lib.format.open_memmap(os.path.join(a.out, f"{a.split}_fg.npy"), "w+", np.float16, (n, 64))
  meta = {k: [] for k in META}
  o = 0
  for s in shards:
    d = np.load(s)
    for k in META:
      meta[k].append(d[k])
    for i in range(0, len(d["image"]), a.bs):
      x = torch.from_numpy(d["image"][i : i + a.bs]).cuda().permute(0, 3, 1, 2).float() / 255
      with torch.autocast("cuda", dtype=torch.bfloat16):
        t = net.forward_features((x - mean) / std)["x_norm_patchtokens"].float()
      b = t.shape[0]
      t = t.view(b, 8, 2, 8, 2, 384).mean(dim=(2, 4)).reshape(b, 64, 384)
      if "mask" in d:
        m = torch.from_numpy(d["mask"][i : i + a.bs]).cuda().float()[:, None]
        m = F.max_pool2d(F.adaptive_avg_pool2d(m, 8), 3, stride=1, padding=1)[:, 0].reshape(b, 64)
      else:  # datasets without masks (SPEED+): uniform weights
        m = torch.ones(b, 64, device="cuda")
      tok[o : o + b] = t.half().cpu().numpy()
      fg[o : o + b] = m.half().cpu().numpy()
      o += b
    print(f"{os.path.basename(s)}: {o}/{n}", flush=True)
  tok.flush()
  fg.flush()
  np.savez(os.path.join(a.out, f"{a.split}_meta.npz"), **{k: np.concatenate(v) for k, v in meta.items()})


if __name__ == "__main__":
  main()
