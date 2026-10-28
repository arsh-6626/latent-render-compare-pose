"""Sanity check of tless_field.TlessGeometry before training: the object's models_eval points at the crop's GT pose
(allocentric R, translation_scaled -> camera frame) projected into the crop must land on its visible mask.

  DATADIR=... PYTHONPATH=src:lib/bop_toolkit python scripts/tless/field_geom_check.py --crops <export_crops.py --meta dir>
"""

import argparse
import glob
import os

import jax.numpy as jnp
import numpy as np
from bop_toolkit_lib import inout

from liepose.exp.jepa_probe.probe_tless import wxyz_to_mat
from liepose.exp.jepa_probe.tless_field import TlessGeometry, tless_crop_meta


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--crops", required=True)
  p.add_argument("--n", type=int, default=300)
  p.add_argument("--model_scale", type=float, default=0.001, help="models_eval units (mm) -> the dataset translation units (m)")
  a = p.parse_args()
  D = os.environ["DATADIR"]
  sh = [np.load(f) for f in sorted(glob.glob(os.path.join(a.crops, "shard_*.npz")))]
  mask = np.concatenate([s["mask"] for s in sh])[: a.n]
  obj = np.concatenate([s["object_id"] for s in sh])[: a.n]
  R = wxyz_to_mat(np.concatenate([s["rotation_allo"] for s in sh])[: a.n])
  k, _, ts = tless_crop_meta(a.crops)
  k, ts = k[: a.n], ts[: a.n]
  Re, t = map(np.asarray, TlessGeometry().cam_pose(jnp.asarray(R, jnp.float32), jnp.asarray(ts), jnp.asarray(k)))
  if "translation" in sh[0].files:
    tt = np.concatenate([s["translation"] for s in sh])[: a.n]
    print("camera t vs dataset translation: median abs diff", np.median(np.abs(t - tt), 0))
  models = {}
  inside, onmask = [], []
  for i in range(a.n):
    o = int(obj[i])
    if o not in models:
      models[o] = inout.load_ply(os.path.join(D, f"bop_datasets/tless/models_eval/obj_{o:06d}.ply"))["pts"] * a.model_scale
    P = models[o] @ Re[i].T + t[i]
    u = P[:, :2] / P[:, 2:3] * k[i, 6:8] + k[i, 8:10]
    c = (u - k[i, 1:3]) * k[i, 0]  # crop px
    ok = (c >= 0).all(1) & (c < 224).all(1)
    inside.append(ok.mean())
    ci = np.clip(c.astype(int), 0, 223)
    onmask.append(mask[i][ci[ok, 1], ci[ok, 0]].mean() if ok.any() else 0.0)
  inside, onmask = np.array(inside), np.array(onmask)
  print(f"t (camera frame) median {np.median(t, 0)}; points inside the crop: median {np.median(inside):.3f}; "
        f"on the visible mask: median {np.median(onmask):.3f}, 10th pct {np.percentile(onmask, 10):.3f}")


if __name__ == "__main__":
  main()
