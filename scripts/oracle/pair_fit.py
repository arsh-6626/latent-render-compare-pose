"""Paired supervision, readouts that separate "too few anchors" from "the network can't represent it".

1. Anchor fit: on training anchors (tet/cube/icosa; for cone/cyl part of rot_off is about the symmetry axis and
   is not an error), observation = the GT render's tokens, candidates = its rendered perturbations.
   P(E_pert > E_gt) by offset bin. Near 1 at 0.5-1 deg means the model fits fine structure where it has targets.
2. Ceiling by anchor distance: the refinement-ceiling pick error (from ceiling_wm.py's *_energy.npy) split by
   the symmetry-aware rotation distance from each test pose to its nearest anchor of the same shape.

  PYTHONPATH=src python scripts/oracle/pair_fit.py --wm $LOGS/probe/v6_pair --pairs $DATADIR/pairs \
      --ceiling $DATADIR/oracle/ceiling.npz --cache $DATADIR/dino_cache --out $DATADIR/oracle/pair_fit_v6_pair.json
"""

import argparse
import json
import os
import pickle
import sys

import flax
import jax
import jax.numpy as jnp
import numpy as np

from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, FrozenPoseModel, pool_tokens, weighted_energy

sys.path.insert(0, os.path.dirname(__file__))
from ceiling_render import wxyz_to_mat  # noqa: E402

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]
ROT_BINS = [(0.1, 0.35), (0.35, 0.75), (0.75, 1.5), (1.5, 3.0), (3.0, 6.1)]
TRAN_BINS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.1)]


def load_wm(path, refs_tok, refs_fg):
  wa = json.load(open(os.path.join(path, "args.json")))
  refs, _ = pool_tokens(refs_tok, refs_fg, wa["pool"])
  model = FrozenPoseModel(dim=wa["dim"], depth=wa["depth"], heads=wa["dim"] // 64, out_dim=refs.shape[-1])
  params = model.init(jax.random.PRNGKey(0), jnp.zeros((1,) + refs.shape[1:]), jnp.zeros((1, 9)))
  params = flax.serialization.from_bytes(params, open(os.path.join(path, "params.msgpack"), "rb").read())
  return model, jax.tree_util.tree_map(jnp.asarray, params), jnp.asarray(refs, jnp.float32), wa["pool"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--wm", required=True)
  p.add_argument("--pairs", required=True)
  p.add_argument("--ceiling", required=True)
  p.add_argument("--cache", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--per_shape", type=int, default=300)
  a = p.parse_args()
  from liepose.exp.jepa_probe.probe import pose_feat

  model, params, refs, pool = load_wm(a.wm, np.load(os.path.join(a.cache, "refs_tokens.npy")),
                                      np.load(os.path.join(a.cache, "refs_fg.npy")))
  meta = np.load(os.path.join(a.pairs, "pairs_meta.npz"))
  tok = np.load(os.path.join(a.pairs, "pairs_tokens.npy"), mmap_mode="r")
  fgs = np.load(os.path.join(a.pairs, "pairs_fg.npy"), mmap_mode="r")
  C = tok.shape[1]

  @jax.jit
  def energy(shape, z, w, g):
    pred = model.apply(params, jnp.broadcast_to(refs[shape], (g.shape[0],) + refs.shape[1:]), g)
    return weighted_energy(pred, z[None], (FG_FLOOR + w)[None])

  # 1. anchor fit (discrete-symmetry shapes only)
  res = {"wm": a.wm, "anchor_fit": {}}
  rng = np.random.default_rng(0)
  win_r, win_t = {b: [] for b in ROT_BINS}, {b: [] for b in TRAN_BINS}
  for si in range(3):
    idx = np.sort(rng.choice(np.nonzero(meta["label"] == si)[0], a.per_shape, replace=False))
    for m in idx:
      z, w = pool_tokens(np.asarray(tok[m, :1], np.float32), np.asarray(fgs[m, :1], np.float32).reshape(1, 16, 16), pool)
      E = np.asarray(energy(si, jnp.asarray(z[0]), jnp.asarray(w[0], jnp.float32),
                            jnp.asarray(pose_feat(meta["R"][m], meta["t"][m]), jnp.float32)))
      for c in range(1, C):
        ro, to = meta["rot_off"][m, c], meta["tran_off"][m, c]
        for lo, hi in ROT_BINS if ro > 0 else []:
          if lo <= ro < hi:
            win_r[(lo, hi)].append(E[c] > E[0])
        for lo, hi in TRAN_BINS if to > 0 else []:
          if lo <= to < hi:
            win_t[(lo, hi)].append(E[c] > E[0])
  res["anchor_fit"]["P(E_pert>E_gt) by rot_off"] = {f"{lo}-{hi}deg": float(np.mean(v)) for (lo, hi), v in win_r.items() if v}
  res["anchor_fit"]["P(E_pert>E_gt) by tran_off"] = {f"{lo}-{hi}": float(np.mean(v)) for (lo, hi), v in win_t.items() if v}

  # 2. ceiling pick error by distance to the nearest anchor
  Cz = np.load(a.ceiling)
  name = os.path.basename(a.wm.rstrip("/"))
  E = np.load(os.path.join(os.path.dirname(a.ceiling), f"ceiling_wm_{name}_energy.npy"))
  eqs = pickle.load(open(os.path.join(a.cache, "test-5k_equivalents.pkl"), "rb"))
  rot = Cz["kind"] == "rot"
  near, pick = [], []
  for i, ti in enumerate(Cz["index"]):
    A = meta["R"][meta["label"] == Cz["label"][i], 0]  # (n_anchor,3,3)
    Req = wxyz_to_mat(eqs[ti])  # (m,3,3)
    tr = np.einsum("aij,mij->am", A, Req)
    near.append(np.degrees(np.arccos(np.clip((tr.max() - 1) / 2, -1, 1))))
    c = np.nonzero(rot[i])[0]
    pick.append(Cz["rot_err"][i, c[np.argmin(E[i, c])]])
  near, pick = np.asarray(near), np.asarray(pick)
  q = np.quantile(near, [0, 1 / 3, 2 / 3, 1])
  res["ceiling_by_anchor_dist"] = {
    f"{q[k]:.1f}-{q[k + 1]:.1f}deg": {"n": int(m.sum()), "pick_err": float(pick[m].mean())}
    for k in range(3) for m in [(near >= q[k]) & (near <= q[k + 1])]}
  res["ceiling_by_anchor_dist"]["all"] = {"n": len(pick), "pick_err": float(pick.mean()),
                                          "median_anchor_dist": float(np.median(near))}
  json.dump(res, open(a.out, "w"), indent=2)
  print(json.dumps(res, indent=2))


if __name__ == "__main__":
  main()
