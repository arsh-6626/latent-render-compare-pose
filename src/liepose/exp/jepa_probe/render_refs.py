"""Render one canonical (identity-pose) reference image per SYMSOL-T shape.

Uses the renderer settings stored in the dataset's config.yaml so the references
match the training images. Needs an X display for open3d (e.g. DISPLAY=:1).

  python -m liepose.exp.jepa_probe.render_refs --data $DATADIR/symsolt/train-45k
"""

import argparse
import os

import numpy as np
from omegaconf import OmegaConf

from liepose.data.symsolt.dataset import SHAPE_NAMES
from liepose.data.symsolt.presets import preset_dict
from liepose.data.symsolt.synth import Synthesizer, load_config


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", type=str, required=True)
  parser.add_argument("--out", type=str, default=None)
  args = parser.parse_args()

  conf = load_config()
  config_path = os.path.join(args.data, "config.yaml")
  if os.path.isfile(config_path):
    conf = OmegaConf.merge(conf, OmegaConf.load(config_path))
  else:
    print(f"warning: {config_path} not found, using synth.py defaults")

  s = Synthesizer(width=conf.width, height=conf.height, hfov=conf.hfov,
                  aspect=conf.aspect, x=conf.x)
  refs, refs_color = [], []
  for name in SHAPE_NAMES:  # index order == label_shape
    preset = preset_dict[name]
    assert preset.obj_index == SHAPE_NAMES.index(name), name
    refs.append(s.render(preset.create(size=conf.shape_size), np.eye(4)))
    refs_color.append(s.render(preset.create(size=conf.shape_size, color=True), np.eye(4)))

  out = args.out or os.path.join(args.data, "refs.npz")
  np.savez(out, none=np.stack(refs), color=np.stack(refs_color), shapes=np.array(SHAPE_NAMES))
  print(f"saved {out}")


if __name__ == "__main__":
  main()
