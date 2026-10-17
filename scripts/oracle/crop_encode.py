"""Zoomed object-centred crops (torch-CUDA env, e.g. `jepa`): each image is cropped to a square around its own
foreground box (like a detector, never the GT pose), side = scale * max(box w, h), resized to 224, and encoded
with DINOv2 (stored 2x2-pooled). In SYMSOL-T the object is ~75 px wide in the 224 image, ~3x3 pooled tokens; the
crop makes it ~8x8, so a sub-degree pose change moves ~3x more feature area.

Per image we store crop = (s, x0, y0): crop pixel = s * (image pixel - (x0, y0)), i.e. crop intrinsics
f' = s f, c' = s (c - x0 | c - y0). Background is black, so padding outside the image is exact.

  python scripts/oracle/crop_encode.py split   --data $DATADIR/symsolt/train-225k --name train-225k --out $DATADIR/crop_cache
  python scripts/oracle/crop_encode.py ceiling --ceiling $DATADIR/oracle/ceiling.npz --out $DATADIR/crop_cache
  python scripts/oracle/crop_encode.py pairs   --pairs $DATADIR/pairs --out $DATADIR/crop_cache/pairs
  python scripts/oracle/crop_encode.py rolls   --data $DATADIR/symsolt/train-225k --rolls $DATADIR/rolls_wide \
      --out $DATADIR/crop_cache/rolls_wide
"""

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import imageio
import numpy as np
import torch
from scipy.ndimage import rotate

from cache_dino import encode, fg_weights, load_annos

SIZE = 224


def crop(img, scale, min_side=32.0):
  """-> (224,224,3) crop, (s, x0, y0)"""
  ys, xs = np.nonzero(img.max(-1) > 10)
  if len(xs) == 0:
    cx = cy = img.shape[0] / 2
    side = float(img.shape[0])
  else:
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = max(scale * max(xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()), min_side)
  s, x0, y0 = SIZE / side, cx - side / 2, cy - side / 2
  M = np.array([[s, 0, -s * x0], [0, s, -s * y0]], np.float64)
  out = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
  return out, np.array([s, x0, y0], np.float32)


class Encoder:
  def __init__(self, pool, scale):
    self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
    self.pool, self.scale, self.g = pool, scale, 16 // pool
    self.ex = ThreadPoolExecutor(16)

  def __call__(self, imgs):
    """(n,H,W,3) uint8 -> pooled tokens (n,g*g,384) fp16, fg (n,g*g) fp16, crop (n,3)"""
    out = list(self.ex.map(lambda im: crop(im, self.scale), imgs))
    c = np.stack([o[0] for o in out])
    n, g, p = len(c), self.g, self.pool
    tok = encode(self.net, c).reshape(n, g, p, g, p, 384).astype(np.float32).mean(axis=(2, 4)).reshape(n, g * g, 384)
    fg = fg_weights(c).reshape(n, g, p, g, p).max(axis=(2, 4)).reshape(n, g * g)
    return tok.astype(np.float16), fg.astype(np.float16), np.stack([o[1] for o in out])


def main():
  p = argparse.ArgumentParser()
  p.add_argument("mode", choices=["split", "ceiling", "pairs", "rolls"])
  p.add_argument("--out", required=True)
  p.add_argument("--data")
  p.add_argument("--name")
  p.add_argument("--ceiling")
  p.add_argument("--pairs")
  p.add_argument("--rolls")
  p.add_argument("--scale", type=float, default=1.25)
  p.add_argument("--pool", type=int, default=2)
  p.add_argument("--chunk", type=int, default=2048)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  enc = Encoder(a.pool, a.scale)
  P = (16 // a.pool) ** 2
  sfx = f"_p{a.pool}"

  if a.mode == "split":
    annos = load_annos(os.path.join(a.data, "annotations.gz"))
    n = len(annos)
    tok = np.lib.format.open_memmap(os.path.join(a.out, f"{a.name}{sfx}_tokens.npy"), "w+", np.float16, (n, P, 384))
    fg = np.lib.format.open_memmap(os.path.join(a.out, f"{a.name}{sfx}_fg.npy"), "w+", np.float16, (n, P))
    crops = np.zeros((n, 3), np.float32)
    for s in range(0, n, a.chunk):
      files = [os.path.join(a.data, x["image"]) for x in annos[s : s + a.chunk]]
      imgs = np.stack(list(enc.ex.map(lambda f: np.asarray(imageio.v2.imread(f))[..., :3], files)))
      tok[s : s + len(files)], fg[s : s + len(files)], crops[s : s + len(files)] = enc(imgs)
      print(f"{a.name}: {s + len(files)}/{n}", flush=True)
    tok.flush()
    fg.flush()
    np.savez(os.path.join(a.out, f"{a.name}{sfx}_meta.npz"),
             rotation=np.array([x["rotation"] for x in annos], np.float64),
             translation=np.array([x["translation"] for x in annos], np.float64),
             label=np.array([x["label_shape"] for x in annos]), crop=crops)

  elif a.mode == "ceiling":
    C = np.load(a.ceiling)
    tok, fg, crops = enc(C["obs"][..., :3])
    np.savez(os.path.join(a.out, "ceiling_obs.npz"), tokens=tok, fg=fg, crop=crops, index=C["index"])
    print(f"ceiling obs: {tok.shape}, mean zoom {crops[:, 0].mean():.2f}x", flush=True)

  else:
    if a.mode == "pairs":
      shards = sorted(glob.glob(os.path.join(a.pairs, "*_[0-9][0-9].npz")))
      meta = np.load(os.path.join(a.pairs, "pairs_meta.npz"))

      def batches():
        for sh in shards:
          d = np.load(sh)
          yield d["images"][..., :3]
    else:
      meta = np.load(os.path.join(a.rolls, "pairs_meta.npz"))
      annos = load_annos(os.path.join(a.data, "annotations.gz"))

      def load(m):
        img = np.asarray(imageio.v2.imread(os.path.join(a.data, annos[meta["index"][m]]["image"])))[..., :3]
        return np.stack([img] + [np.clip(rotate(img, -th, reshape=False, order=1), 0, 255).astype(np.uint8)
                                 for th in meta["roll_deg"][m, 1:]])

      def batches():
        for s in range(0, len(meta["index"]), 256):
          yield np.stack(list(enc.ex.map(load, range(s, min(s + 256, len(meta["index"]))))))
    M, C = meta["R"].shape[:2]
    tok = np.lib.format.open_memmap(os.path.join(a.out, "pairs_tokens.npy"), "w+", np.float16, (M, C, P, 384))
    fg = np.lib.format.open_memmap(os.path.join(a.out, "pairs_fg.npy"), "w+", np.float16, (M, C, P))
    crops = np.zeros((M, C, 3), np.float32)
    o = 0
    for imgs in batches():
      n = len(imgs)
      t_, f_, c_ = enc(imgs.reshape(n * C, *imgs.shape[2:]))
      tok[o : o + n], fg[o : o + n], crops[o : o + n] = t_.reshape(n, C, P, 384), f_.reshape(n, C, P), c_.reshape(n, C, 3)
      o += n
      print(f"{a.mode}: {o}/{M}", flush=True)
    tok.flush()
    fg.flush()
    np.savez(os.path.join(a.out, "pairs_meta.npz"), **{k: meta[k] for k in meta.files}, crop=crops)


if __name__ == "__main__":
  main()
