"""Refinement ceiling, step 3: the same fine-perturbation candidates scored by a trained world model's latent
energy E(g) = ||P(z_ref, g) - z_obs||^2 instead of a distance to an exact render. Comparable to
ceiling_feats.py (same candidates, same metrics).

  PYTHONPATH=src python scripts/oracle/ceiling_wm.py --wm $LOGS/probe/v4_trans --cache $DATADIR/dino_cache \
      --ceiling $DATADIR/oracle/ceiling.npz --out $DATADIR/oracle/ceiling_wm_v4.json
"""

import argparse
import json
import os
import sys

import flax
import jax
import jax.numpy as jnp
import numpy as np

from liepose.exp.jepa_probe.probe import pose_feat
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, FrozenPoseModel, pool_tokens, weighted_energy

sys.path.insert(0, os.path.dirname(__file__))
from ceiling_feats import summarize  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--wm", required=True)
  p.add_argument("--cache", required=True)
  p.add_argument("--ceiling", required=True)
  p.add_argument("--out", required=True)
  a = p.parse_args()
  wa = json.load(open(os.path.join(a.wm, "args.json")))
  pool = wa["pool"]
  C = np.load(a.ceiling)
  refs, _ = pool_tokens(np.load(os.path.join(a.cache, "refs_tokens.npy")), np.load(os.path.join(a.cache, "refs_fg.npy")), pool)
  model = FrozenPoseModel(dim=wa["dim"], depth=wa["depth"], heads=wa["dim"] // 64, out_dim=refs.shape[-1])
  params = model.init(jax.random.PRNGKey(0), jnp.zeros((1,) + refs.shape[1:]), jnp.zeros((1, 9)))
  params = flax.serialization.from_bytes(params, open(os.path.join(a.wm, "params.msgpack"), "rb").read())
  params = jax.tree_util.tree_map(jnp.asarray, params)
  refs = jnp.asarray(refs, jnp.float32)

  idx = C["index"]
  tok = np.load(os.path.join(a.cache, "test-5k_tokens.npy"), mmap_mode="r")
  fg = np.load(os.path.join(a.cache, "test-5k_fg.npy"), mmap_mode="r")
  z_obs, w_obs = pool_tokens(np.asarray(tok[idx]), np.asarray(fg[idx]), pool)

  @jax.jit
  def energy(shape, z, w, g):
    pred = model.apply(params, jnp.broadcast_to(refs[shape], (g.shape[0],) + refs.shape[1:]), g)
    return weighted_energy(pred, z.astype(jnp.float32)[None], (FG_FLOOR + w)[None])

  E = np.stack([np.asarray(energy(int(C["label"][i]), z_obs[i], w_obs[i].astype(np.float32),
                                  jnp.asarray(pose_feat(C["R"][i], C["t"][i]), jnp.float32))) for i in range(len(idx))])
  np.save(a.out.replace(".json", "_energy.npy"), E)
  r = summarize(E, C["rot_err"], C["tran_err"], C["kind"], C["label"], ["tet", "cube", "icosa", "cone", "cyl"])
  json.dump(r, open(a.out, "w"), indent=2)
  print(f"== world model {a.wm} (pool {pool})")
  print("  P(E_pert > E_gt) by sym-aware rot err: " + "  ".join(f"{b}: {v:.3f}" for b, v in r["P(d_pert>d_gt) by rot err"].items() if v is not None))
  print("  P(E_pert > E_gt) by translation offset: " + "  ".join(f"{b}: {v:.3f}" for b, v in r["P(d_pert>d_gt) by tran offset"].items()))
  s = r["select_rot_err"]
  print(f"  pick among perturbed (0.25-8deg): argmin_E {s['argmin_d']:.2f} deg | best {s['best_candidate']:.2f} | "
        f"random {s['random']:.2f} | per shape " + " ".join(f"{k}={v:.2f}" for k, v in s["per_shape_argmin_d"].items()))


if __name__ == "__main__":
  main()
