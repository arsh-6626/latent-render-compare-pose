"""Refinement ceiling, step 1: render fine pose perturbations of SYMSOL-T test samples.

For each selected test image, render the object at its GT pose and at poses perturbed by small rotations
(0.25..8 deg, random axes, GT translation) and small translations (GT rotation). With an exact renderer (the
FoundationPose setting) the GT render equals the observation, so the question is purely whether a feature
space can resolve these perturbations. Symmetry-aware errors are stored per candidate (rotations about a
symmetry axis are not errors).

  DISPLAY=:1 PYTHONPATH=src python scripts/oracle/ceiling_render.py --data $DATADIR/symsolt/test-5k \
      --cache $DATADIR/dino_cache --out $DATADIR/oracle/ceiling.npz
"""

import argparse
import os
import pickle
import sys

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from liepose.data.symsolt.dataset import SHAPE_NAMES
from liepose.data.symsolt.presets import preset_dict
from liepose.data.symsolt.synth import Synthesizer, load_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from cache_dino import load_annos  # noqa: E402

ROT_DEG = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
TRAN = [0.005, 0.01, 0.02, 0.05, 0.1]


def wxyz_to_mat(q):
  q = np.asarray(q, np.float64)
  return Rotation.from_quat(np.roll(q, -1, axis=-1)).as_matrix()


def sym_err_deg(R, R_eq):
  tr = np.einsum("ij,mij->m", R, R_eq)
  return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min())


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True)
  p.add_argument("--cache", required=True, help="dino_cache dir (for test-5k_equivalents.pkl)")
  p.add_argument("--out", required=True)
  p.add_argument("--per_shape", type=int, default=100)
  p.add_argument("--axes", type=int, default=4, help="random axes per rotation level")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  rng = np.random.default_rng(a.seed)

  conf = OmegaConf.merge(load_config(), OmegaConf.load(os.path.join(a.data, "config.yaml")))
  s = Synthesizer(width=conf.width, height=conf.height, hfov=conf.hfov, aspect=conf.aspect, x=conf.x)
  annos = load_annos(os.path.join(a.data, "annotations.gz"))
  eqs = pickle.load(open(os.path.join(a.cache, "test-5k_equivalents.pkl"), "rb"))
  labels = np.array([x["label_shape"] for x in annos])

  out = {k: [] for k in ["index", "label", "obs", "renders", "R", "t", "rot_err", "tran_err", "kind", "level",
                         "obs_vs_gt_render"]}
  for si, name in enumerate(SHAPE_NAMES):
    mesh = preset_dict[name].create(size=conf.shape_size)
    idx = np.sort(rng.choice(np.nonzero(labels == si)[0], a.per_shape, replace=False))
    for i in tqdm(idx, desc=name):
      R0, t0 = wxyz_to_mat(annos[i]["rotation"]), np.asarray(annos[i]["translation"], np.float64)
      R_eq = wxyz_to_mat(eqs[i])
      cands = [(R0, t0, "gt", 0.0)]
      for d in ROT_DEG:
        for _ in range(a.axes):
          ax = rng.normal(size=3)
          ax /= np.linalg.norm(ax)
          cands.append((Rotation.from_rotvec(ax * np.radians(d)).as_matrix() @ R0, t0, "rot", d))
      for d in TRAN:
        for _ in range(2):
          v = rng.normal(size=3)
          cands.append((R0, t0 + d * v / np.linalg.norm(v), "tran", d))
      imgs = []
      for R, t, _, _ in cands:
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R, t
        imgs.append(s.render(mesh, pose)[..., :3])
      obs = imageio.imread(os.path.join(a.data, annos[i]["image"]))[..., :3]
      out["index"].append(i)
      out["label"].append(si)
      out["obs"].append(obs)
      out["renders"].append(np.stack(imgs))
      out["R"].append(np.stack([c[0] for c in cands]))
      out["t"].append(np.stack([c[1] for c in cands]))
      out["rot_err"].append([sym_err_deg(c[0], R_eq) for c in cands])
      out["tran_err"].append([float(np.linalg.norm(c[1] - t0)) for c in cands])
      out["kind"].append([c[2] for c in cands])
      out["level"].append([c[3] for c in cands])
      out["obs_vs_gt_render"].append(float(np.abs(obs.astype(np.int16) - imgs[0].astype(np.int16)).mean()))
  out = {k: np.asarray(v) for k, v in out.items()}
  np.savez_compressed(a.out, **out)
  print(f"saved {a.out}: {out['renders'].shape}; mean |obs - render(GT)| = {out['obs_vs_gt_render'].mean():.3f} "
        f"(max {out['obs_vs_gt_render'].max():.3f})")


if __name__ == "__main__":
  main()
