"""Paired supervision for the world model, step 1: render SYMSOL-T training poses and fine perturbations of them.

The world model is otherwise only ever supervised at the GT pose of each image (nearest training neighbour is
~4 deg away), so it never sees what the object looks like at p + delta. For each selected training sample this
renders the object at its GT pose and at small rotation / translation offsets, giving token targets at every
candidate (encoded by pair_encode.py, used by probe_frozen.py --pairs). Training split only; the refinement
ceiling test (ceiling_render.py) uses test-5k.

  DISPLAY=:1 PYTHONPATH=src python scripts/oracle/pair_render.py --data $DATADIR/symsolt/train-45k \
      --shape tet --out $DATADIR/pairs
"""

import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from liepose.data.symsolt.dataset import SHAPE_NAMES
from liepose.data.symsolt.presets import preset_dict
from liepose.data.symsolt.synth import Synthesizer, load_config

sys.path.insert(0, os.path.dirname(__file__))
from cache_dino import load_annos  # noqa: E402
from ceiling_render import wxyz_to_mat  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True)
  p.add_argument("--shape", required=True, choices=SHAPE_NAMES)
  p.add_argument("--out", required=True)
  p.add_argument("--per_shape", type=int, default=2000)
  p.add_argument("--n_rot", type=int, default=6, help="rotation offsets per pose (log-uniform angle, random axis)")
  p.add_argument("--n_tran", type=int, default=2, help="translation offsets per pose (log-uniform norm)")
  p.add_argument("--rot_deg", type=float, nargs=2, default=[0.1, 6.0])
  p.add_argument("--tran", type=float, nargs=2, default=[0.002, 0.08])
  p.add_argument("--shard", type=int, default=500)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  si = SHAPE_NAMES.index(a.shape)
  rng = np.random.default_rng(a.seed + si)
  os.makedirs(a.out, exist_ok=True)

  conf = OmegaConf.merge(load_config(), OmegaConf.load(os.path.join(a.data, "config.yaml")))
  s = Synthesizer(width=conf.width, height=conf.height, hfov=conf.hfov, aspect=conf.aspect, x=conf.x)
  annos = load_annos(os.path.join(a.data, "annotations.gz"))
  labels = np.array([x["label_shape"] for x in annos])
  mesh = preset_dict[a.shape].create(size=conf.shape_size)
  idx = np.sort(rng.choice(np.nonzero(labels == si)[0], a.per_shape, replace=False))

  for k, lo in enumerate(range(0, len(idx), a.shard)):
    path = os.path.join(a.out, f"{a.shape}_{k:02d}.npz")
    if os.path.isfile(path):
      continue
    out = {k_: [] for k_ in ["index", "R", "t", "rot_off", "tran_off", "images"]}
    for i in idx[lo : lo + a.shard]:
      R0, t0 = wxyz_to_mat(annos[i]["rotation"]), np.asarray(annos[i]["translation"], np.float64)
      Rs, ts, ro, to = [R0], [t0], [0.0], [0.0]
      for _ in range(a.n_rot):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        d = np.exp(rng.uniform(*np.log(a.rot_deg)))
        Rs.append(Rotation.from_rotvec(ax * np.radians(d)).as_matrix() @ R0)
        ts.append(t0)
        ro.append(d)
        to.append(0.0)
      for _ in range(a.n_tran):
        v = rng.normal(size=3)
        d = np.exp(rng.uniform(*np.log(a.tran)))
        Rs.append(R0)
        ts.append(t0 + d * v / np.linalg.norm(v))
        ro.append(0.0)
        to.append(d)
      imgs = []
      for R, t in zip(Rs, ts):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R, t
        imgs.append(s.render(mesh, pose)[..., :3])
      out["index"].append(i)
      out["R"].append(np.stack(Rs))
      out["t"].append(np.stack(ts))
      out["rot_off"].append(ro)
      out["tran_off"].append(to)
      out["images"].append(np.stack(imgs))
    out = {k_: np.asarray(v) for k_, v in out.items()}
    np.savez_compressed(path, label=np.full(len(out["index"]), si), **out)
    print(f"{a.shape}: saved {path} {out['images'].shape}", flush=True)


if __name__ == "__main__":
  main()
