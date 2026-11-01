"""Port BOP dataset-dict caches built locally to another data root (e.g. the HPC scratch), so jobs there skip the
hours-long annotation pass (one mask PNG read per instance, slow on Lustre). Reads the raw local cache, rewrites every
path under the local data root to the target root, and writes it under the key the target computes (base_loader's
hash_info with the target's resolved paths; the target paths must contain no symlinks).
  python scripts/hpc/port_dict_cache.py --src $DATADIR --dst /scratch/.../liepose/data --out out/bop_cache tless-train_pbr
"""
import argparse, os

from liepose.data.bop.data_utils.common import load_assets_cache, write_assets_cache
from liepose.data.bop.dataset_loaders.base_loader import load_dataset_by_name
from liepose.data.bop.registry import get_dataset, get_dataset_type
import liepose.exp.bop.dataset_registry  # noqa: F401  registers under DATADIR


def rewrite(x, a, b):
  if isinstance(x, str):
    return b + x[len(a):] if x.startswith(a) else x
  if isinstance(x, dict):
    return {k: rewrite(v, a, b) for k, v in x.items()}
  if isinstance(x, list):
    return [rewrite(v, a, b) for v in x]
  return x


def key(info, root, name, cache_root):  # the loader's own __init__ (its defaults), with the loading steps stubbed out
  move = lambda x: x if x is None else os.path.join(root, os.path.relpath(x, os.environ["DATADIR"])) + ("/" if x.endswith("/") else "")
  cls = get_dataset_type(info["dataset_type"])
  saved = {m: getattr(cls, m) for m in STUB}
  try:
    for m in STUB:
      setattr(cls, m, lambda self: None)
    os.environ["CACHE_ROOT_PATH"] = cache_root
    return load_dataset_by_name(name, path=move(info["path"]), models_info_path=move(info["models_info_path"]),
                                test_bop_path=move(info.get("test_bop_path"))).cache_path
  finally:
    for m, f in saved.items():
      setattr(cls, m, f)


STUB = ("_load_models_info", "_load_dataset_dicts", "_load_detections", "_filter_objects")
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--out", required=True)
ap.add_argument("names", nargs="+")
a = ap.parse_args()
src, dst, src_cache = os.path.realpath(a.src), a.dst.rstrip("/"), os.environ["CACHE_ROOT_PATH"]
for name in a.names:
  info = get_dataset(name)
  inp = key(info, src, name, src_cache)
  dicts = load_assets_cache(inp)
  assert dicts is not None, inp
  dicts = rewrite(dicts, src + "/", dst + "/")
  assert not any(str(d.get("file_name", "")).startswith(src) for d in dicts), name
  out = key(info, dst, name, a.out)
  write_assets_cache(out, dicts)
  print(name, len(dicts), "dicts", inp, "->", out, "| e.g.", dicts[0]["file_name"], dicts[0].get("depth_path"))
