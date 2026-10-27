"""Is there a subspace of frozen DINOv2 tokens where pose signal beats the feature field's render error? (diagnostic)

Closed form, no training: a per-token head can only select or discard directions of each 384-d token, so if no
direction has signal well above the render error, no per-token head (feat_field.py) can remove the render
bottleneck.
  render error  e = field(g*) - z on held-out val crops (field.py checkpoint, GT pose)    -> S_e = E[e e^T] (per token)
  pose signal   d = z(rot(x, base + 1 deg)) - z(rot(x, base)) on val crops, blur-matched, the frozen DINOv2 flax port
                (an exact in-plane roll of the image; the signal roll_oracle.py's oracle arm sees) -> S_s = E[d d^T]
  generalised eigenproblem S_s v = lam S_e v: lam = (1 deg signal / render error)^2 along v; isotropic reference
  tr S_s / tr S_e. Writes the eigenvectors (S_e-whitened, descending lam) for roll_oracle.py --proj.

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit:scripts/speedplus python scripts/speedplus/renderable_subspace.py \
      --field $LOGS/probe/sp_field_v1 --out <npz>
"""

import argparse
import glob
import json
import os

import flax
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from liepose.exp.jepa_probe.comparator import load_field
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import crop_meta
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.models.dinov2 import DinoV2, pool2
from roll_oracle import rotate


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--field", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--n_err", type=int, default=4000)
  p.add_argument("--n_sig", type=int, default=2000)
  p.add_argument("--base", type=float, default=30.0)
  p.add_argument("--deg", type=float, default=1.0)
  p.add_argument("--split", default="train", help="crops the subspace is estimated on (train: no overlap with the val tests)")
  a = p.parse_args()
  D = os.environ["DATADIR"]
  fm, fp = load_field(a.field)
  va = TokenSplit(os.path.join(D, "speedplus_tokens"), a.split)
  meta, idx, _ = crop_meta(os.path.join(D, "speedplus_crops", a.split))
  assert (idx == va.index).all()

  render = jax.jit(lambda R, t, k, z: fm.apply(fp, None, None, None, None, R, t, k, app_stats(z)))
  Se, n = np.zeros((384, 384)), 0
  for s in range(0, a.n_err, 100):
    r = np.arange(s, min(s + 100, a.n_err))
    z = jnp.asarray(va.tokens[r], jnp.float32)
    e = np.asarray(render(jnp.asarray(va.R[r], jnp.float32), jnp.asarray(va.t[r], jnp.float32), jnp.asarray(meta[r]), z) - z, np.float64)
    e = e.reshape(-1, 384)
    Se += e.T @ e
    n += len(e)
  Se /= n

  dino = DinoV2()
  dp = jax.tree_util.tree_map(jnp.asarray, flax.serialization.msgpack_restore(open(os.path.join(D, "dinov2_vits14_224.msgpack"), "rb").read()))
  rot = jax.vmap(rotate)
  enc = jax.jit(lambda x, th: pool2(dino.apply(dp, rot(x, th))))
  ims = []
  for f in sorted(glob.glob(os.path.join(D, "speedplus_crops", a.split, "shard_*.npz"))):
    ims.append(np.load(f)["image"][: a.n_sig - sum(len(x) for x in ims)])
    if sum(len(x) for x in ims) >= a.n_sig:
      break
  ims = np.concatenate(ims)
  Ss, m = np.zeros((384, 384)), 0
  for s in range(0, len(ims), 50):
    x = jnp.asarray(ims[s:s + 50], jnp.float32)
    b = len(x)
    d = np.asarray(enc(x, jnp.full(b, a.base + a.deg)) - enc(x, jnp.full(b, a.base)), np.float64).reshape(-1, 384)
    Ss += d.T @ d
    m += len(d)
  Ss /= m

  eps = 1e-6 * np.trace(Se) / 384
  lam, V = scipy.linalg.eigh(Ss, Se + eps * np.eye(384))  # V^T S_e V = I
  o = np.argsort(lam)[::-1]
  lam, V = lam[o], V[:, o]
  iso = float(np.trace(Ss) / np.trace(Se))
  cum = lambda k: float(lam[:k].sum() / k)
  res = {"split": a.split, "field": a.field, "n_err_tokens": int(n), "n_sig_tokens": int(m), "deg": a.deg, "isotropic_ratio": iso,
         "lam_top10": [float(x) for x in lam[:10]], "lam_mean_top": {str(k): cum(k) for k in (1, 4, 16, 64, 128)},
         "n_lam_gt1": int((lam > 1).sum()), "n_lam_gt_iso": int((lam > iso).sum())}
  print(json.dumps(res, indent=1))
  np.savez(a.out, lam=lam, V=V.astype(np.float32), Se=Se, Ss=Ss)
  json.dump(res, open(a.out.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
  main()
