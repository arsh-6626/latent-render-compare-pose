"""Render the oracle rotation bank: per shape, N_bank uniform rotations + N_q query rotations, t=0.

Used to test whether an encoder's latent is rotation-sensitive with no predictor involved:
rank bank images by ||f(x_bank) - f(x_query)|| and check whether the nearest ones lie on the
query's symmetry-equivalence set.

  DISPLAY=:1 PYTHONPATH=src python scripts/oracle/render_bank.py --config $DATADIR/symsolt/test-5k/config.yaml --out $DATADIR/oracle/bank.npz
"""

import argparse

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from liepose.data.symsolt.dataset import SHAPE_NAMES
from liepose.data.symsolt.presets import preset_dict
from liepose.data.symsolt.synth import Synthesizer, load_config


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--config", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--n_bank", type=int, default=2048)
  p.add_argument("--n_query", type=int, default=20)
  p.add_argument("--seed", type=int, default=7)
  a = p.parse_args()

  conf = OmegaConf.merge(load_config(), OmegaConf.load(a.config))
  s = Synthesizer(width=conf.width, height=conf.height, hfov=conf.hfov, aspect=conf.aspect, x=conf.x)

  R_bank = Rotation.random(a.n_bank, random_state=a.seed).as_matrix()
  R_query = Rotation.random(a.n_query, random_state=a.seed + 1).as_matrix()
  R_all = np.concatenate([R_bank, R_query])

  images, eq_list = [], []
  for name in SHAPE_NAMES:
    preset = preset_dict[name]
    mesh = preset.create(size=conf.shape_size)
    imgs = []
    for R in tqdm(R_all, desc=name):
      pose = np.eye(4)
      pose[:3, :3] = R
      imgs.append(s.render(mesh, pose)[..., :3])
    images.append(np.stack(imgs))
    for R in R_query:
      pose = np.eye(4)
      pose[:3, :3] = R
      eq_list.append(preset.get_symmetric_poses(pose)[:, :3, :3])

  max_eq = max(len(e) for e in eq_list)
  R_eq = np.zeros((len(SHAPE_NAMES), a.n_query, max_eq, 3, 3))
  n_eq = np.zeros((len(SHAPE_NAMES), a.n_query), dtype=np.int64)
  for k, e in enumerate(eq_list):
    si, qi = divmod(k, a.n_query)
    R_eq[si, qi, : len(e)] = e
    n_eq[si, qi] = len(e)

  np.savez_compressed(a.out, images=np.stack(images).astype(np.uint8), R_bank=R_bank,
                      R_query=R_query, R_eq=R_eq, n_eq=n_eq, shapes=np.array(SHAPE_NAMES))
  print(f"saved {a.out}")


if __name__ == "__main__":
  main()
