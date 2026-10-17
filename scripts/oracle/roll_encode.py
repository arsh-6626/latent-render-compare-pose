"""Free fine supervision from in-plane rolls (torch-CUDA env, e.g. `jepa`): no renderer needed.

Rotating the image by -theta about the principal point is exactly the pose (Rz(theta) R, Rz(theta) t) for a pinhole
camera with the principal point at the image centre (checked against re-renders: mean |diff| 0.5 vs 3.6 unrolled at
5 deg; SYMSOL-T background is black, so the corners are exact too). Each selected train-225k image gives candidate 0 =
the image itself and n_roll rolled copies with their exact poses, in the pairs format probe_frozen.PairSet reads
(tokens stored already 2x2-pooled; `index` indexes train-225k).

  python scripts/oracle/roll_encode.py --data $DATADIR/symsolt/train-225k --out $DATADIR/rolls
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import imageio
import numpy as np
import torch
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation

from cache_dino import encode, fg_weights, load_annos


def wxyz_to_mat(q):
  return Rotation.from_quat(np.roll(np.asarray(q, np.float64), -1, axis=-1)).as_matrix()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--per_shape", type=int, default=2000)
  p.add_argument("--n_roll", type=int, default=4)
  p.add_argument("--deg", type=float, nargs=2, default=[0.25, 8.0], help="log-uniform |theta|, random sign")
  p.add_argument("--wide", action="store_true", help="last two rolls wide (+-30, 180+-30 deg), the rest small")
  p.add_argument("--pool", type=int, default=2)
  p.add_argument("--seed", type=int, default=1)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  rng = np.random.default_rng(a.seed)
  annos = load_annos(os.path.join(a.data, "annotations.gz"))
  labels = np.array([x["label_shape"] for x in annos])
  idx = np.sort(np.concatenate([rng.choice(np.nonzero(labels == s)[0], a.per_shape, replace=False)
                                for s in np.unique(labels)]))
  M, C, g = len(idx), 1 + a.n_roll, 16 // a.pool
  theta = np.zeros((M, C))
  theta[:, 1:] = np.exp(rng.uniform(*np.log(a.deg), size=(M, a.n_roll))) * rng.choice([-1, 1], size=(M, a.n_roll))
  if a.wide and a.n_roll >= 4:  # FastPose-ViT ranges for the last two: U(-30, 30) and U(150, 210)
    theta[:, -2] = rng.uniform(-30, 30, size=M)
    theta[:, -1] = rng.uniform(150, 210, size=M)
  R0 = wxyz_to_mat([annos[i]["rotation"] for i in idx])
  t0 = np.array([annos[i]["translation"] for i in idx], np.float64)
  Rz = Rotation.from_rotvec(np.stack([np.zeros_like(theta), np.zeros_like(theta), np.radians(theta)], -1).reshape(-1, 3)
                            ).as_matrix().reshape(M, C, 3, 3)
  R = Rz @ R0[:, None]
  t = np.einsum("mcij,mj->mci", Rz, t0)

  net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
  tok = np.lib.format.open_memmap(os.path.join(a.out, "pairs_tokens.npy"), "w+", np.float16, (M, C, g * g, 384))
  fg = np.lib.format.open_memmap(os.path.join(a.out, "pairs_fg.npy"), "w+", np.float16, (M, C, g * g))
  ex = ThreadPoolExecutor(16)

  def load(m):
    img = np.asarray(imageio.v2.imread(os.path.join(a.data, annos[idx[m]]["image"])))[..., :3]
    return np.stack([img] + [np.clip(rotate(img, -th, reshape=False, order=1), 0, 255).astype(np.uint8)
                             for th in theta[m, 1:]])

  for s in range(0, M, 512):
    imgs = np.concatenate(list(ex.map(load, range(s, min(s + 512, M)))))
    n = len(imgs) // C
    t_ = encode(net, imgs).reshape(n * C, g, a.pool, g, a.pool, 384).astype(np.float32).mean(axis=(2, 4))
    f_ = fg_weights(imgs).reshape(n * C, g, a.pool, g, a.pool).max(axis=(2, 4))
    tok[s : s + n] = t_.reshape(n, C, g * g, 384).astype(np.float16)
    fg[s : s + n] = f_.reshape(n, C, g * g).astype(np.float16)
    print(f"{s + n}/{M}", flush=True)
  tok.flush()
  fg.flush()
  np.savez(os.path.join(a.out, "pairs_meta.npz"), index=idx, label=labels[idx], R=R, t=t,
           rot_off=np.abs(theta), tran_off=np.zeros((M, C)), roll_deg=theta)


if __name__ == "__main__":
  main()
