"""Cache frozen DINOv2 patch tokens + patch-level foreground weights for SYMSOL-T (torch env, e.g. `jepa`).

Writes, per split, <out>/<split>_tokens.npy (N, 256, 384) fp16, <split>_fg.npy (N, 16, 16) fp16, and
<split>_meta.npz (rotation wxyz, translation, label, image index). Also caches the canonical refs.

  python scripts/oracle/cache_dino.py --data $DATADIR/symsolt --out $DATADIR/dino_cache
"""

import argparse
import gzip
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


def load_annos(path):
  # liepose.data.utils.load_file writes gzip'd pickle or json depending on version; try both
  with gzip.open(path, "rb") as f:
    raw = f.read()
  try:
    return pickle.loads(raw)
  except Exception:
    return json.loads(raw.decode())


def fg_weights(images):
  """(B,H,W,3) uint8 -> (B,16,16) fraction of object pixels per patch, dilated by one patch."""
  m = torch.from_numpy((images.max(axis=-1) > 10).astype(np.float32))[:, None]
  frac = F.adaptive_avg_pool2d(m, 16)
  dil = F.max_pool2d(frac, 3, stride=1, padding=1)
  return dil[:, 0].numpy()


@torch.no_grad()
def encode(net, images, bs=256):
  mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
  out = []
  for i in range(0, len(images), bs):
    x = torch.from_numpy(images[i : i + bs]).cuda().permute(0, 3, 1, 2).float() / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
      f = net.forward_features((x - mean) / std)["x_norm_patchtokens"]
    out.append(f.float().half().cpu().numpy())
  return np.concatenate(out)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--splits", nargs="+", default=["test-5k", "train-45k"])
  p.add_argument("--chunk", type=int, default=4096)
  p.add_argument("--pool", type=int, default=1, help="store tokens average-pooled by this factor (fg max-pooled)")
  p.add_argument("--skip_refs", action="store_true")
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()

  if not a.skip_refs:
    refs = np.load(os.path.join(a.data, "test-5k", "refs.npz"))["none"][..., :3]
    np.save(os.path.join(a.out, "refs_tokens.npy"), encode(net, refs))
    np.save(os.path.join(a.out, "refs_fg.npy"), fg_weights(refs).astype(np.float16))
  g = 16 // a.pool
  sfx = "" if a.pool == 1 else f"_p{a.pool}"

  def pool(tok, fgw):
    if a.pool == 1:
      return tok, fgw
    n = len(tok)
    tok = tok.reshape(n, g, a.pool, g, a.pool, -1).astype(np.float32).mean(axis=(2, 4)).astype(np.float16)
    return tok.reshape(n, g * g, -1), fgw.reshape(n, g, a.pool, g, a.pool).max(axis=(2, 4)).reshape(n, g * g)

  for split in a.splits:
    root = os.path.join(a.data, split)
    annos = load_annos(os.path.join(root, "annotations.gz"))
    n = len(annos)
    tok = np.lib.format.open_memmap(os.path.join(a.out, f"{split}{sfx}_tokens.npy"), "w+", np.float16, (n, g * g, 384))
    fg = np.lib.format.open_memmap(os.path.join(a.out, f"{split}{sfx}_fg.npy"), "w+", np.float16, (n, g, g) if a.pool == 1 else (n, g * g))
    ex = ThreadPoolExecutor(16)
    for s in range(0, n, a.chunk):
      files = [os.path.join(root, x["image"]) for x in annos[s : s + a.chunk]]
      imgs = np.stack(list(ex.map(lambda f: imageio.imread(f)[..., :3], files)))
      t_, f_ = pool(encode(net, imgs), fg_weights(imgs))
      tok[s : s + len(files)] = t_
      fg[s : s + len(files)] = f_
      print(f"{split}: {s + len(files)}/{n}", flush=True)
    tok.flush()
    fg.flush()
    np.savez(os.path.join(a.out, f"{split}{sfx}_meta.npz"),
             rotation=np.array([x["rotation"] for x in annos], np.float64),
             translation=np.array([x["translation"] for x in annos], np.float64),
             label=np.array([x["label_shape"] for x in annos]))
    if split.startswith("test"):
      with open(os.path.join(a.out, f"{split}_equivalents.pkl"), "wb") as f:
        pickle.dump([np.array(x["rotations_equivalent"]).reshape(-1, 4) for x in annos], f)


if __name__ == "__main__":
  main()
