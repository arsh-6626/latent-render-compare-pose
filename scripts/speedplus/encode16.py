"""SPEED+ crop shards -> unpooled 16 x 16 frozen DINOv2 tokens (flax port, liepose.models.dinov2; = the hub model to
1e-5), in the row order of the 8 x 8 cache (speedplus_tokens), whose meta / fg layout it copies: <out>/<split>_tokens.npy
(N, 256, 384) fp16, <split>_fg.npy ones (N, 256) (SPEED+ has no masks), <split>_meta.npz (= the 8 x 8 cache's).

  DATADIR=... PYTHONPATH=src python scripts/speedplus/encode16.py --splits train val val_aug --out $DATADIR/speedplus_tokens16
"""

import argparse
import glob
import os
import shutil

import flax
import jax
import jax.numpy as jnp
import numpy as np

from liepose.models.dinov2 import DinoV2


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--splits", nargs="+", default=["train", "val", "val_aug"])
  p.add_argument("--out", required=True)
  p.add_argument("--bs", type=int, default=64)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  os.makedirs(a.out, exist_ok=True)
  dino = DinoV2()
  dp = jax.tree_util.tree_map(jnp.asarray, flax.serialization.msgpack_restore(open(os.path.join(D, "dinov2_vits14_224.msgpack"), "rb").read()))
  enc = jax.jit(lambda x: dino.apply(dp, x).astype(jnp.float16))
  for split in a.splits:
    ref = os.path.join(D, "speedplus_tokens")
    m = np.load(os.path.join(ref, f"{split}_meta.npz"))
    n = len(m["object_id"])
    tok = np.lib.format.open_memmap(os.path.join(a.out, f"{split}_tokens.npy"), "w+", np.float16, (n, 256, 384))
    o = 0
    for f in sorted(glob.glob(os.path.join(D, "speedplus_crops", split, "shard_*.npz"))):
      d = np.load(f)
      assert (d["index"] == m["index"][o:o + len(d["index"])]).all(), f"{f}: row order differs from the 8x8 cache"
      im = d["image"]
      for i in range(0, len(im), a.bs):
        tok[o + i:o + i + len(im[i:i + a.bs])] = np.asarray(enc(jnp.asarray(im[i:i + a.bs], jnp.float32)))
      o += len(im)
      print(f"{split} {os.path.basename(f)}: {o}/{n}", flush=True)
    assert o == n
    tok.flush()
    np.save(os.path.join(a.out, f"{split}_fg.npy"), np.ones((n, 256), np.float16))
    shutil.copy(os.path.join(ref, f"{split}_meta.npz"), os.path.join(a.out, f"{split}_meta.npz"))


if __name__ == "__main__":
  main()
