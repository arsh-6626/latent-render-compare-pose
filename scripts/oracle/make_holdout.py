"""SYMSOL-T world-model validation split, held out from train-225k (no test-5k image is used for any choice).

Holds out --frac of the training images per shape, among images that are not anchors of the rendered pairs or the
rolls (so those targets come from kept images only), and writes for each cache dir
  {trainhold}_p2_{tokens,fg,meta}  the kept images (world-model training split and reference bank)
  {valhold}_p2_{tokens,fg,meta}    the held-out images
and a ceiling set on --per_shape held-out images per shape with the candidate recipe of ceiling_render.py
(GT + 4 random axes x 0.25..8 deg, GT translation; 2 directions x 0.005..0.1 translation), poses only:
  val_ceiling.npz       candidates + symmetric errors
  val_ceiling_obs.npz   observation tokens / fg (/ crop) of those images, per cache dir

  PYTHONPATH=src python scripts/oracle/make_holdout.py --data $DATADIR/symsolt/train-225k \
      --caches $DATADIR/crop_cache --pairs $DATADIR/crop_cache/pairs $DATADIR/crop_cache/rolls_wide \
      --out $DATADIR/oracle
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from cache_dino import load_annos  # noqa: E402
from ceiling_render import ROT_DEG, TRAN, sym_err_deg, wxyz_to_mat  # noqa: E402


def write_split(cache, src, name, rows):
  tok = np.load(os.path.join(cache, f"{src}_p2_tokens.npy"), mmap_mode="r")
  fg = np.load(os.path.join(cache, f"{src}_p2_fg.npy"), mmap_mode="r")
  meta = dict(np.load(os.path.join(cache, f"{src}_p2_meta.npz")))
  out = np.lib.format.open_memmap(os.path.join(cache, f"{name}_p2_tokens.npy"), "w+", tok.dtype, (len(rows),) + tok.shape[1:])
  for i in range(0, len(rows), 8192):
    out[i : i + 8192] = tok[rows[i : i + 8192]]
  out.flush()
  np.save(os.path.join(cache, f"{name}_p2_fg.npy"), np.asarray(fg[rows]))
  np.savez(os.path.join(cache, f"{name}_p2_meta.npz"), source_row=rows, **{k: v[rows] for k, v in meta.items()})
  print(f"{cache}: {name} {len(rows)} rows", flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True, help="symsolt train-225k dir (annotations.gz: symmetric equivalents)")
  p.add_argument("--caches", nargs="+", required=True)
  p.add_argument("--pairs", nargs="*", default=[], help="pair / roll dirs whose anchors stay in training")
  p.add_argument("--out", required=True)
  p.add_argument("--src", default="train-225k")
  p.add_argument("--train_name", default="trainhold")
  p.add_argument("--val_name", default="valhold")
  p.add_argument("--frac", type=float, default=0.03)
  p.add_argument("--per_shape", type=int, default=100)
  p.add_argument("--axes", type=int, default=4)
  p.add_argument("--seed", type=int, default=1234)
  a = p.parse_args()
  rng = np.random.default_rng(a.seed)

  meta = np.load(os.path.join(a.caches[0], f"{a.src}_p2_meta.npz"))
  label = meta["label"]
  anchors = np.unique(np.concatenate([np.load(os.path.join(d, "pairs_meta.npz"))["index"] for d in a.pairs] or [np.zeros(0, int)]))
  free = np.setdiff1d(np.arange(len(label)), anchors)
  val = np.sort(np.concatenate([rng.choice(free[label[free] == s], int(round(a.frac * (label == s).sum())), replace=False)
                                for s in np.unique(label)]))
  train = np.setdiff1d(np.arange(len(label)), val)
  assert not np.isin(anchors, val).any()
  print(f"held out {len(val)} of {len(label)} ({np.bincount(label[val])} per shape); {len(anchors)} pair/roll anchors kept",
        flush=True)
  for c in a.caches:
    m = np.load(os.path.join(c, f"{a.src}_p2_meta.npz"))
    assert (m["label"] == label).all() and np.allclose(m["rotation"], meta["rotation"]), f"{c} misaligned"
    write_split(c, a.src, a.train_name, train)
    write_split(c, a.src, a.val_name, val)

  # ceiling candidates on held-out images (same recipe as ceiling_render.py, no renders)
  annos = load_annos(os.path.join(a.data, "annotations.gz"))
  assert len(annos) == len(label)
  pos = {r: k for k, r in enumerate(val)}  # source row -> row in the val split
  out = {k: [] for k in ["index", "source_row", "label", "R", "t", "rot_err", "tran_err", "kind", "level"]}
  for s in np.unique(label):
    for i in np.sort(rng.choice(val[label[val] == s], a.per_shape, replace=False)):
      x = annos[i]
      assert x["label_shape"] == s
      R0, t0 = wxyz_to_mat(x["rotation"]), np.asarray(x["translation"], np.float64)
      assert np.allclose(R0, wxyz_to_mat(meta["rotation"][i]), atol=1e-5)
      R_eq = wxyz_to_mat(np.array(x["rotations_equivalent"]).reshape(-1, 4))
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
      out["index"].append(pos[i])
      out["source_row"].append(i)
      out["label"].append(s)
      out["R"].append(np.stack([c[0] for c in cands]))
      out["t"].append(np.stack([c[1] for c in cands]))
      out["rot_err"].append([sym_err_deg(c[0], R_eq) for c in cands])
      out["tran_err"].append([float(np.linalg.norm(c[1] - t0)) for c in cands])
      out["kind"].append([c[2] for c in cands])
      out["level"].append([c[3] for c in cands])
  out = {k: np.asarray(v) for k, v in out.items()}
  os.makedirs(a.out, exist_ok=True)
  np.savez_compressed(os.path.join(a.out, "val_ceiling.npz"), **out)
  for c in a.caches:
    tok = np.load(os.path.join(c, f"{a.val_name}_p2_tokens.npy"), mmap_mode="r")
    fg = np.load(os.path.join(c, f"{a.val_name}_p2_fg.npy"))
    m = np.load(os.path.join(c, f"{a.val_name}_p2_meta.npz"))
    obs = {"index": out["index"], "tokens": np.asarray(tok[out["index"]]), "fg": fg[out["index"]]}
    if "crop" in m.files:
      obs["crop"] = m["crop"][out["index"]]
    np.savez(os.path.join(c, "val_ceiling_obs.npz"), **obs)
  print(f"val ceiling: {len(out['index'])} obs x {out['R'].shape[1]} candidates; GT rot_err max {out['rot_err'][:, 0].max():.2e}")


if __name__ == "__main__":
  main()
