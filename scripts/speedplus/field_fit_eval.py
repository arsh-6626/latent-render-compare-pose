"""Feature field with test-time appearance fitting (NeRF-W style; field.py --lat_dim) on the SPEED+ synthetic val split.

For each observation and each candidate pose g, the energy is E_fit(g) = min_l |render(g, l) - z_obs|^2 + lat_l2 |l|^2
over the per-image appearance latent l (--fit_iters Adam steps from l = 0; the same freedom for every candidate, so
the ranking stays fair), against E_0(g) with l = 0 (the statistics code only). Inference reads the observation's
tokens and box meta only (GT: the candidate sets and the scorer). Candidates: GT + --per rotations at 0.25-1 / 1-3 /
3-10 deg and an estimate 1-3 deg away (wm_eval.Eval's bins). Reported: P(E_pert > E_gt) per bin, GT error / d_app.

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit:scripts/speedplus python scripts/speedplus/field_fit_eval.py \
      --field $LOGS/probe/sp_field_lat --out <json>
"""

import argparse
import json
import os
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from liepose.exp.jepa_probe.field import FeatureField, FieldModel
from liepose.exp.jepa_probe.probe_frozen import perturb
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import SpeedPlusGeometry, crop_meta
from liepose.exp.jepa_probe.wm_eval import app_distance, app_stats

BINS = [(0.25, 1.0), (1.0, 3.0), (3.0, 10.0)]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--field", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--split", default="val")
  p.add_argument("--n", type=int, default=300)
  p.add_argument("--per", type=int, default=6)
  p.add_argument("--fit_iters", type=int, default=40)
  p.add_argument("--fit_lr", type=float, default=0.05)
  p.add_argument("--batch", type=int, default=4, help="observations per call (x candidates renders)")
  a = p.parse_args()
  D = os.environ["DATADIR"]
  fa = SimpleNamespace(**json.load(open(os.path.join(a.field, "args.json"))))
  assert fa.lat_dim > 0, "needs a field trained with --lat_dim"
  model = FieldModel(field=FeatureField(plane_res=fa.plane_res, plane_dim=fa.plane_dim, hid=fa.hid, radius=fa.radius, n_samp=fa.n_samp,
                                        lat_dim=fa.lat_dim, geo=SpeedPlusGeometry()))
  obs = TokenSplit(os.path.join(D, "speedplus_tokens"), a.split)
  meta, idx, _ = crop_meta(os.path.join(D, "speedplus_crops", a.split))
  assert (idx == obs.index).all()
  p0 = model.init(jax.random.PRNGKey(0), None, None, None, None, jnp.eye(3)[None], jnp.zeros((1, 3)), jnp.asarray(meta[:1]), jnp.zeros((1, 768)))
  params = jax.tree_util.tree_map(jnp.asarray, flax.serialization.from_bytes(p0, open(os.path.join(a.field, "params.msgpack"), "rb").read()))
  d_app = app_distance(D)
  rng = np.random.default_rng(7)
  rows = np.sort(rng.choice(len(obs.R), a.n, replace=False))
  tx = optax.adam(a.fit_lr)

  @jax.jit
  def energies(R, t, k, z):
    """R (M, 3, 3), t (M, 3), k (M, 6), z (M, 64, 384) -> E_0 (M,), E_fit (M,)"""
    app = app_stats(z)

    def E(lat):
      pred = model.apply(params, None, None, None, None, R, t, k, app, lat=lat)
      return ((pred - z) ** 2).mean((-1, -2))

    lat = jnp.zeros((R.shape[0], fa.lat_dim))
    e0 = E(lat)

    def body(c, _):
      lat, s = c
      g = jax.grad(lambda l: (E(l) + fa.lat_l2 * (l ** 2).sum(-1)).sum())(lat)
      u, s = tx.update(g, s, lat)
      return (optax.apply_updates(lat, u), s), None

    (lat, _), _ = jax.lax.scan(body, (lat, tx.init(lat)), None, a.fit_iters)
    return e0, E(lat)

  kinds = ["gt"] + sum([[f"{lo:g}-{hi:g}"] * a.per for lo, hi in BINS], []) + ["est"]
  kinds = np.array(kinds)
  E0, Ef = [], []
  for s in range(0, a.n, a.batch):
    r = rows[s:s + a.batch]
    n = len(r)
    R = obs.R[r]
    Rs = [R[:, None]] + [perturb(np.repeat(R, a.per, 0), hi, lo, rng).reshape(n, a.per, 3, 3) for lo, hi in BINS] + [perturb(R, 3.0, 1.0, rng)[:, None]]
    Rc = np.concatenate(Rs, 1)
    C = Rc.shape[1]
    rep = lambda x: np.repeat(x, C, 0)
    e0, ef = energies(jnp.asarray(Rc.reshape(-1, 3, 3), jnp.float32), jnp.asarray(rep(obs.t[r]), jnp.float32), jnp.asarray(rep(meta[r])),
                      jnp.asarray(rep(obs.tokens[r]), jnp.float32))
    E0.append(np.asarray(e0).reshape(n, C))
    Ef.append(np.asarray(ef).reshape(n, C))
    if (s // a.batch) % 25 == 0:
      print(f"{s + n}/{a.n}", flush=True)
  E0, Ef = np.concatenate(E0), np.concatenate(Ef)
  res = {"field": a.field, "split": a.split, "n": a.n, "fit_iters": a.fit_iters}
  for name, E in [("no_fit", E0), ("fit", Ef)]:
    r = {f"P {k}": float((E[:, kinds == k] > E[:, :1]).mean()) for k in [f"{lo:g}-{hi:g}" for lo, hi in BINS] + ["est"]}
    r["gt_err_over_app"] = float(np.median(np.sqrt(E[:, 0])) / d_app)
    res[name] = r
    print(name, json.dumps(r), flush=True)
  json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
