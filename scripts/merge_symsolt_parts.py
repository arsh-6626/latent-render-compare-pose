"""Merge per-shape SYMSOL-T synth runs (one process per shape) into one split directory.

Each part dir <parts>/<shape>/ contains <shape>/images, <shape>/points.ply and annotations.gz whose
image paths are already prefixed with the shape name. The merged split symlinks each shape dir.

  PYTHONPATH=src python scripts/merge_symsolt_parts.py --parts $DATADIR/symsolt/train-225k-parts --out $DATADIR/symsolt/train-225k
"""

import argparse
import os
import shutil

from liepose.data.utils import load_file, write_file

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--parts", required=True)
  p.add_argument("--out", required=True)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  annos = []
  for sh in SHAPES:
    part = os.path.join(a.parts, sh)
    part_annos = load_file(os.path.join(part, "annotations.gz"))
    assert all(x["image"].startswith(sh + "/") for x in part_annos), sh
    annos += part_annos
    link = os.path.join(a.out, sh)
    if not os.path.islink(link):
      os.symlink(os.path.join(part, sh), link)
    print(sh, len(part_annos))
  write_file(os.path.join(a.out, "annotations.gz"), annos)
  shutil.copy(os.path.join(a.parts, SHAPES[0], "config.yaml"), os.path.join(a.out, "config.yaml"))
  print("total", len(annos))


if __name__ == "__main__":
  main()
