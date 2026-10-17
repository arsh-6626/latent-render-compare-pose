"""Paired supervision, step 2: DINOv2 tokens for the renders from pair_render.py (torch-CUDA env, e.g. `jepa`).

Writes <pairs>/pairs_tokens.npy (M,C,256,384) fp16, pairs_fg.npy (M,C,256) fp16 and pairs_meta.npz
(index, label, R, t, rot_off, tran_off), with C candidates per pose (candidate 0 = GT render).

  python scripts/oracle/pair_encode.py --pairs $DATADIR/pairs
"""

import argparse
import glob
import os

import numpy as np
import torch

from cache_dino import encode, fg_weights

META = ["index", "label", "R", "t", "rot_off", "tran_off"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--pairs", required=True)
  a = p.parse_args()
  shards = sorted(glob.glob(os.path.join(a.pairs, "*_[0-9][0-9].npz")))
  sizes = [np.load(s)["index"].shape[0] for s in shards]
  C = np.load(shards[0])["R"].shape[1]
  M = sum(sizes)
  net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
  tok = np.lib.format.open_memmap(os.path.join(a.pairs, "pairs_tokens.npy"), "w+", np.float16, (M, C, 256, 384))
  fg = np.lib.format.open_memmap(os.path.join(a.pairs, "pairs_fg.npy"), "w+", np.float16, (M, C, 256))
  meta = {k: [] for k in META}
  o = 0
  for s, n in zip(shards, sizes):
    d = np.load(s)
    imgs = d["images"].reshape(n * C, *d["images"].shape[2:])
    tok[o : o + n] = encode(net, imgs).reshape(n, C, 256, 384)
    fg[o : o + n] = fg_weights(imgs).reshape(n, C, 256).astype(np.float16)
    for k in META:
      meta[k].append(d[k])
    o += n
    print(f"{os.path.basename(s)}: {o}/{M}", flush=True)
  tok.flush()
  fg.flush()
  np.savez(os.path.join(a.pairs, "pairs_meta.npz"), **{k: np.concatenate(v) for k, v in meta.items()})


if __name__ == "__main__":
  main()
