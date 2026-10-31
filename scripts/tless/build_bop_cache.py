"""Build a BOP dataset-dicts cache (the one the loader writes on its first use) with one process per scene, then join the
scenes in the loader's own order and write it to the loader's default cache path. Same records as the serial build; the
serial one takes ~11 min per YCB-V scene. YCB-V loader only (its `scenes` option).

  DATADIR=... CACHE_ROOT_PATH=... PYTHONPATH=src python scripts/tless/build_bop_cache.py ycbv-train_real --procs 12
"""

import argparse
import copy
import os
from multiprocessing import Pool

import liepose.exp.bop.dataset_registry  # noqa: F401  (registers the datasets)
from liepose.data.bop.data_utils import get_default_cache_path, load_assets_cache, write_assets_cache
from liepose.data.bop.dataset_loaders.base_loader import load_dataset_by_name
from liepose.data.bop.registry import get_dataset


def one_scene(arg):
  name, scene, tmp = arg
  path = os.path.join(tmp, f"{scene}.gz")
  if not os.path.isfile(path):
    load_dataset_by_name(name, scenes=[scene], cache_path=path)
  return path


def main():
  p = argparse.ArgumentParser()
  p.add_argument("name")
  p.add_argument("--procs", type=int, default=12)
  a = p.parse_args()
  info = copy.deepcopy(get_dataset(a.name))
  # the loader's default cache path (base_loader.BOPDatasetLoader.__init__, default width / height)
  hash_info = [a.name, info["path"], info["models_info_path"], info.get("test_bop_path"), info.get("width", 640),
               info.get("height", 480)]
  out = get_default_cache_path(a.name, "dataset_dicts", hash_info)
  if os.path.isfile(out):
    print(f"exists: {out}")
    return
  scenes = sorted(os.listdir(info["path"]))
  tmp = os.path.join(os.path.dirname(out), "_scenes")  # resumable: finished scenes are kept until the join
  os.makedirs(tmp, exist_ok=True)
  with Pool(a.procs) as pool:
    parts = pool.map(one_scene, [(a.name, s, tmp) for s in scenes], chunksize=1)
  dicts = []
  for f in parts:
    dicts.extend(load_assets_cache(f))
  write_assets_cache(out, dicts)
  for f in parts:
    os.remove(f)
  os.rmdir(tmp)
  print(f"{a.name}: {len(scenes)} scenes, {len(dicts)} dataset dicts -> {out}")


if __name__ == "__main__":
  main()
