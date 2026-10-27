"""Pose refinement by minimising the feature field's energy E(g) = mean |render(g) - z_obs|^2 (field.py, frozen) on
the SPEED+ synthetic validation split (full split; no lightbox / sunlamp image).

  A  bias (diagnostic, starts AT the true pose): gradient descent on E from the GT pose; how far the minimum of E
     sits from the truth (the floor for any refiner on this energy). Labelled diagnostic, GT-initialised.
  B  pipeline (GT-free): LiePose's 16-sample consensus (select_wm2.py's rule: most samples within 10 deg) ->
     CEM (--cem_iters x --cem_pop, Gaussian in so(3) x translation, elites --cem_elite) -> Adam on E; official
     ESA (esa_score.esa) of the consensus pick and of the refined pose. Inference reads the crop tokens and box meta
     only; GT is read by the scorer only.
Pose: LiePose's allocentric rotation and translation_scaled (the field's inputs), R <- exp(w) R, t <- t + dt.

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit:scripts/speedplus python scripts/speedplus/field_refine.py \
      --field $LOGS/probe/sp_field_v1 --samples <dir>/val_samples.npz --out <json>
"""

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esa_score import allo_to_ego, esa  # noqa: E402

from liepose.data.speedplus.dataset import recover_translation  # noqa: E402
from liepose.exp.jepa_probe.field import FeatureField, FieldModel  # noqa: E402
from liepose.exp.jepa_probe.probe_tless import TokenSplit  # noqa: E402
from liepose.exp.jepa_probe.sp_geometry import SpeedPlusGeometry, crop_meta  # noqa: E402
from liepose.exp.jepa_probe.wm_eval import app_stats  # noqa: E402
from liepose.exp.speedplus.run import TRAN_OFFSET, TRAN_SCALE  # noqa: E402


def expm(w):
  """(N, 3) rotation vectors -> (N, 3, 3)"""
  th = jnp.sqrt((w ** 2).sum(-1, keepdims=True) + 1e-12)[..., None]  # smooth at w = 0 (the GD start)
  k = w / th[..., 0]
  K = jnp.stack([jnp.zeros_like(k[:, 0]), -k[:, 2], k[:, 1], k[:, 2], jnp.zeros_like(k[:, 0]), -k[:, 0], -k[:, 1], k[:, 0],
                 jnp.zeros_like(k[:, 0])], -1).reshape(-1, 3, 3)
  return jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * K @ K


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--field", required=True)
  p.add_argument("--samples", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--batch", type=int, default=128)
  p.add_argument("--gd_iters", type=int, default=100)
  p.add_argument("--lr_rot", type=float, default=2e-3, help="Adam step on the rotation vector (rad)")
  p.add_argument("--lr_t", type=float, default=2e-3, help="Adam step on translation_scaled")
  p.add_argument("--cem_iters", type=int, default=5)
  p.add_argument("--cem_pop", type=int, default=64)
  p.add_argument("--cem_elite", type=int, default=8)
  p.add_argument("--cem_rot_deg", type=float, default=3.0)
  p.add_argument("--cem_t", type=float, default=0.02)
  p.add_argument("--limit", type=int, default=0)
  p.add_argument("--cem_chunk", type=int, default=16, help="observations per CEM energy call (x cem_pop renders)")
  a = p.parse_args()
  D = os.environ["DATADIR"]
  fa = SimpleNamespace(**json.load(open(os.path.join(a.field, "args.json"))))
  field = FeatureField(plane_res=fa.plane_res, plane_dim=fa.plane_dim, hid=getattr(fa, "hid", 128), radius=fa.radius, n_samp=fa.n_samp,
                       sub=2 * getattr(fa, "grid", 8), geo=SpeedPlusGeometry())
  model = FieldModel(field=field)
  obs = TokenSplit(os.path.join(D, getattr(fa, "tokens", "speedplus_tokens")), "val")  # the field's own token space
  meta, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", "val"))
  assert (idx == obs.index).all()
  p0 = model.init(jax.random.PRNGKey(0), None, None, None, None, jnp.eye(3)[None], jnp.zeros((1, 3)), jnp.asarray(meta[:1]), jnp.zeros((1, 768)))
  params = jax.tree_util.tree_map(jnp.asarray, flax.serialization.from_bytes(p0, open(os.path.join(a.field, "params.msgpack"), "rb").read()))

  @jax.jit
  def energy(R, t, k, z):  # (N, C, ...) candidates of N observations -> (N, C)
    N, C = R.shape[:2]
    rep = lambda x: jnp.repeat(x, C, 0)
    pred = model.apply(params, None, None, None, None, R.reshape(N * C, 3, 3), t.reshape(N * C, 3), rep(k), rep(app_stats(z)))
    return ((pred.reshape(N, C, *z.shape[1:]) - z[:, None]) ** 2).mean((-1, -2))

  tx = optax.multi_transform({"w": optax.adam(a.lr_rot), "t": optax.adam(a.lr_t)}, {"w": "w", "t": "t"})

  @jax.jit
  def gd(R, t, k, z):
    def loss(v):
      return energy(expm(v["w"])[:, None] @ R[:, None], (t + v["t"])[:, None], k, z).sum()

    v = {"w": jnp.zeros_like(t), "t": jnp.zeros_like(t)}

    def body(c, _):
      v, s = c
      g = jax.grad(loss)(v)
      u, s = tx.update(g, s, v)
      return (optax.apply_updates(v, u), s), None

    (v, _), _ = jax.lax.scan(body, (v, tx.init(v)), None, a.gd_iters)
    return expm(v["w"]) @ R, t + v["t"]

  def cem(R, t, k, z, key):
    N = R.shape[0]
    mu_w, mu_t = jnp.zeros((N, 3)), jnp.zeros((N, 3))
    sd_w, sd_t = jnp.full((N, 3), np.radians(a.cem_rot_deg)), jnp.full((N, 3), a.cem_t)
    for _ in range(a.cem_iters):
      key, k1, k2 = jax.random.split(key, 3)
      w = mu_w[:, None] + sd_w[:, None] * jax.random.normal(k1, (N, a.cem_pop, 3))
      dt = mu_t[:, None] + sd_t[:, None] * jax.random.normal(k2, (N, a.cem_pop, 3))
      w, dt = w.at[:, 0].set(mu_w), dt.at[:, 0].set(mu_t)  # keep the current mean in the population
      Rc = expm(w.reshape(-1, 3)).reshape(N, a.cem_pop, 3, 3) @ R[:, None]
      tt = t[:, None] + dt
      c = a.cem_chunk
      E = jnp.concatenate([energy(Rc[i:i + c], tt[i:i + c], k[i:i + c], z[i:i + c]) for i in range(0, N, c)])
      el = jnp.argsort(E, 1)[:, : a.cem_elite]
      we, te = jnp.take_along_axis(w, el[..., None], 1), jnp.take_along_axis(dt, el[..., None], 1)
      mu_w, mu_t, sd_w, sd_t = we.mean(1), te.mean(1), we.std(1) + 1e-4, te.std(1) + 1e-5
    return expm(mu_w) @ R, t + mu_t

  n = len(obs.R) if not a.limit else a.limit
  sm = np.load(a.samples)
  pos = {int(i): r for r, i in enumerate(sm["index"])}
  srow = np.array([pos[int(i)] for i in obs.index[:n]])
  # consensus pick (GT-free, as select_wm2.py): pairwise geodesic distance between the egocentric samples
  Re_s = sm["R"][srow]
  G = np.degrees(np.arccos(np.clip((np.einsum("nsij,ntij->nst", Re_s, Re_s) - 1) / 2, -1, 1)))
  cons = ((G < 10).sum(-1) - 1e-3 * G.mean(-1)).argmax(1)
  qa = sm["quat_allo"][srow, cons]
  R0 = Rotation.from_quat(np.roll(qa, -1, -1)).as_matrix().astype(np.float32)
  t0 = sm["t_scaled"][srow, cons].astype(np.float32)

  outs = {k: [] for k in ["A_R", "A_t", "B_cem_R", "B_cem_t", "B_R", "B_t"]}
  key = jax.random.PRNGKey(0)
  for s in range(0, n, a.batch):
    r = np.arange(s, min(s + a.batch, n))
    k, z = jnp.asarray(meta[r]), jnp.asarray(obs.tokens[r], jnp.float32)
    Ra, ta = gd(jnp.asarray(obs.R[r], jnp.float32), jnp.asarray(obs.t[r], jnp.float32), k, z)  # A: from GT (diagnostic)
    key, kk = jax.random.split(key)
    Rc, tc = cem(jnp.asarray(R0[r]), jnp.asarray(t0[r]), k, z, kk)  # B: from consensus (GT-free)
    Rb, tb = gd(Rc, tc, k, z)
    for name, v in [("A_R", Ra), ("A_t", ta), ("B_cem_R", Rc), ("B_cem_t", tc), ("B_R", Rb), ("B_t", tb)]:
      outs[name].append(np.asarray(v, np.float64))
    if (s // a.batch) % 20 == 0:
      print(f"{r[-1] + 1}/{n}", flush=True)
  outs = {k: np.concatenate(v) for k, v in outs.items()}

  # scorer only
  sh = [np.load(f) for f in sorted(glob.glob(os.path.join(D, "speedplus_crops", "val", "shard_*.npz")))]
  box = {k: np.concatenate([x[k] for x in sh])[:n] for k in ["bbox", "bbox_center", "focal_ratio", "rotation", "translation"]}
  R_gt = Rotation.from_quat(np.roll(box["rotation"], -1, -1)).as_matrix()
  t_gt = box["translation"].astype(np.float64)
  ang = lambda X, Y: np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", X, Y) - 1) / 2, -1, 1)))
  res = {"field": a.field, "n": int(n), "args": vars(a)}
  dA = ang(outs["A_R"], obs.R[:n])
  res["A_bias_from_gt (diagnostic)"] = {"rot_drift_deg_median": float(np.median(dA)), "rot_drift_deg_mean": float(dA.mean()),
                                        "rot_drift_gt1": float((dA > 1).mean()),
                                        "t_scaled_drift_median": float(np.median(np.linalg.norm(outs["A_t"] - obs.t[:n], axis=-1)))}
  print("A", json.dumps(res["A_bias_from_gt (diagnostic)"]), flush=True)

  def score(Rall, tsc):
    t = recover_translation(tsc, box["bbox"], box["bbox_center"], box["focal_ratio"], TRAN_SCALE, TRAN_OFFSET)
    R = allo_to_ego(Rall, t)
    r = esa(R, t, R_gt, t_gt)
    e = ang(R, R_gt)
    r.update(e_q_gt1=float((e > 1).mean()), e_q_gt10=float((e > 10).mean()))
    return r

  for name, (Rk, tk) in [("B_consensus", (R0[:n].astype(np.float64), t0[:n].astype(np.float64))),
                         ("B_cem", (outs["B_cem_R"], outs["B_cem_t"])), ("B_cem_gd", (outs["B_R"], outs["B_t"]))]:
    res[name] = score(Rk, tk)
    print(name, json.dumps(res[name]), flush=True)
  json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
