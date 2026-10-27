"""World model as a learned 3D feature field of the SPEED+ spacecraft (no retrieval, no CAD model).

Why (wm2.py, v1-v4): every retrieval world model renders a pose from other training images; v4 removed the pull to
the references' pose (shared P est 0.42 -> 0.54) but its error at the true pose stays 0.52x the appearance distance,
twice the token change of a 1 deg rotation (lighting, shadows and background of the borrowed views), so 1-3 deg
stays at chance (0.55). Here the object is one learned model fitted to all ~96k posed training crops:

  field        F(x) for x in the object frame (a box of --radius around the object origin): tri-plane features
               (3 planes, --plane_res^2 x --plane_dim, bilinear) -> MLP -> density sigma(x) and a latent h(x)
  render(g)    the observation's crop (box meta k) and the candidate pose g = (allocentric R, translation_scaled)
               -> camera-frame pose (sp_geometry.cam_pose, pinhole); a ray through each of 16 x 16 sub-patch centres,
               --n_samp depths in [z_c - radius, z_c + radius] (z_c: the object centre's depth), volume rendering of
               h, then per ray a head (h_ray, view direction in the object frame, appearance) -> 384-d DINOv2
               feature; rays that leave the object see a background feature from the appearance code; 2 x 2 average
               pooling -> 8 x 8 tokens, as the token caches
  appearance   from the observation's own token statistics (wm_eval.app_stats: per-channel mean / std over tokens), the
               same for every candidate pose of an observation (it cannot rank poses)
  loss         reconstruction of each training crop's tokens at its GT pose (+ optional pose contrast, --w_nce)

The geometry is exact by construction: a 1 deg rotation moves every ray's hits by exactly 1 deg; nothing is borrowed
from another image. --radius (1.0 m) is a fixed box size, not the keypoint model.
Eval: wm_eval.Eval on the synthetic validation splits, unchanged (the references it retrieves are ignored, so the
shared and per-candidate protocols coincide). Gate, fixed before training (val): P 1-3 >= 0.9, P est >= 0.9, GT
error <= 0.4x the appearance distance; checkpoint choice on val P 1-3.

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src python -m liepose.exp.jepa_probe.field --out $LOGS/probe/sp_field_v1
"""

import argparse
import json
import os
import time
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from liepose.exp.jepa_probe import probe_ret_tless as prt
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import SpeedPlusGeometry, crop_meta
from liepose.exp.jepa_probe.wm_eval import Bank, Eval, app_distance, app_stats


def fourier(x, n):
  f = 2.0 ** jnp.arange(n) * jnp.pi
  return jnp.concatenate([x, jnp.sin(x[..., None] * f).reshape(*x.shape[:-1], -1), jnp.cos(x[..., None] * f).reshape(*x.shape[:-1], -1)], -1)


def bilinear(plane, uv):
  """plane (R, R, C), uv (..., 2) in [-1, 1] -> (..., C); zero outside"""
  R = plane.shape[0]
  p = (uv + 1.0) * 0.5 * (R - 1)
  p0 = jnp.floor(p)
  w = p - p0
  i0 = jnp.clip(p0.astype(jnp.int32), 0, R - 1)
  i1 = jnp.clip(i0 + 1, 0, R - 1)
  g = lambda i, j: plane[i[..., 1], j[..., 0]]
  out = (g(i0, i0) * ((1 - w[..., 0]) * (1 - w[..., 1]))[..., None] + g(i0, i1) * (w[..., 0] * (1 - w[..., 1]))[..., None]
         + g(i1, i0) * ((1 - w[..., 0]) * w[..., 1])[..., None] + g(i1, i1) * (w[..., 0] * w[..., 1])[..., None])
  return out


class FeatureField(nn.Module):
  plane_res: int = 128
  plane_dim: int = 32
  hid: int = 128
  out_dim: int = 384
  radius: float = 1.0
  n_samp: int = 32
  sub: int = 16  # rays per side; pooled to 8 x 8 tokens
  lat_dim: int = 0  # per-image appearance latent (NeRF-W style; --lat_dim), added to the statistics code
  geo: Any = None

  def setup(self):
    init = nn.initializers.normal(0.1)
    self.planes = [self.param(f"plane_{n}", init, (self.plane_res, self.plane_res, self.plane_dim)) for n in ("xy", "xz", "yz")]
    self.mlp = [nn.Dense(self.hid), nn.Dense(self.hid)]
    self.sigma = nn.Dense(1)
    self.app = nn.Dense(64)
    self.film = nn.Dense(2 * self.hid, kernel_init=nn.initializers.zeros)
    self.head = [nn.Dense(256), nn.Dense(256)]
    self.out = nn.Dense(self.out_dim)
    self.bg = nn.Dense(self.out_dim)
    if self.lat_dim:
      self.lat = nn.Dense(64, use_bias=False)  # latent 0 = the statistics code alone

  def point(self, x):
    """x (..., 3) object frame / radius -> sigma (...), h (..., hid)"""
    f = sum(bilinear(p, x[..., ij]) for p, ij in zip(self.planes, ([0, 1], [0, 2], [1, 2])))
    h = jnp.concatenate([f, fourier(x, 4)], -1)
    for d in self.mlp:
      h = nn.gelu(d(h))
    inside = (jnp.abs(x) <= 1.0).all(-1)
    return jnp.where(inside, nn.softplus(self.sigma(h)[..., 0] - 1.0), 0.0), h

  def __call__(self, Rg, tg, kg, app, jitter=None, with_opacity=False, lat=None):
    """candidate poses (N, 3, 3) allocentric / (N, 3) translation_scaled, crop meta (N, 6), appearance statistics
    (N, 2D) -> predicted tokens (N, 64, out_dim); jitter (N, rays, n_samp) in [0, 1): stratified depth samples"""
    geo, S, M = self.geo, self.sub, self.n_samp
    R, t = geo.cam_pose(Rg, tg, kg)  # egocentric, metres
    cc = (jnp.arange(S) + 0.5) * (224.0 / S)
    pc = jnp.stack(jnp.meshgrid(cc, cc, indexing="xy"), -1).reshape(-1, 2)  # crop px, row-major like the tokens
    u = geo.from_crop(jnp.broadcast_to(pc, t.shape[:-1] + pc.shape), kg)  # image px
    d = jnp.concatenate([(u - geo.c) / geo.f, jnp.ones(u.shape[:-1] + (1,))], -1)  # (N, S^2, 3) ray at camera z = 1
    off = (jnp.arange(M) + (0.5 if jitter is None else jitter)) / M  # (M,) or (N, S^2, M)
    z = t[:, None, None, 2] + self.radius * (2.0 * off - 1.0)  # depth samples
    z = jnp.broadcast_to(z, d.shape[:2] + (M,))
    P = d[:, :, None] * z[..., None]  # (N, S^2, M, 3) camera frame
    X = jnp.einsum("nji,nrmj->nrmi", R, P - t[:, None, None]) / self.radius  # object frame / radius
    sigma, h = self.point(X)
    dz = 2.0 * self.radius / M * jnp.linalg.norm(d, axis=-1)[..., None]
    alpha = 1.0 - jnp.exp(-sigma * dz)
    T = jnp.cumprod(jnp.concatenate([jnp.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1] + 1e-10], -1), -1)
    w = T * alpha
    opa = w.sum(-1, keepdims=True)
    h_ray = (w[..., None] * h).sum(-2) / (opa + 1e-6)  # (N, S^2, hid): expected latent of the surface the ray hits
    bgw = 1.0 - opa
    a = self.app(app)
    if self.lat_dim:
      a = a + self.lat(jnp.zeros(app.shape[:-1] + (self.lat_dim,)) if lat is None else lat)
    a = nn.gelu(a)[:, None]  # (N, 1, 64)
    sc, sh = jnp.split(self.film(a), 2, -1)
    vd = jnp.einsum("nji,nrj->nri", R, d / jnp.linalg.norm(d, axis=-1, keepdims=True))  # view direction, object frame
    y = jnp.concatenate([h_ray * (1.0 + sc) + sh, fourier(vd, 2), jnp.broadcast_to(a, h_ray.shape[:2] + (64,))], -1)
    for l in self.head:
      y = nn.gelu(l(y))
    f = (1.0 - bgw) * self.out(y) + bgw * self.bg(a)
    N, D = f.shape[0], f.shape[-1]
    pool = lambda x: x.reshape(N, S // 2, 2, S // 2, 2, x.shape[-1]).mean((2, 4)).reshape(N, (S // 2) ** 2, x.shape[-1])
    if with_opacity:  # (tokens, per-token rendered opacity (N, 64))
      return pool(f), pool(opa)[..., 0]
    return pool(f)


class FieldModel(nn.Module):
  """wm_eval.Eval's model signature (zr, Rr, tr, kr, Rg, tg, kg, app); the references are ignored"""
  field: Any = None

  def __call__(self, zr, Rr, tr, kr, Rg, tg, kg, app, jitter=None, with_opacity=False, lat=None):
    return self.field(Rg, tg, kg, app, jitter, with_opacity, lat)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--steps", type=int, default=60000)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--lr", type=float, default=2e-3)
  p.add_argument("--radius", type=float, default=1.0)
  p.add_argument("--plane_res", type=int, default=128)
  p.add_argument("--plane_dim", type=int, default=32)
  p.add_argument("--n_samp", type=int, default=32)
  p.add_argument("--hid", type=int, default=128)
  p.add_argument("--lat_dim", type=int, default=0, help="per-training-image appearance latent (0: statistics code only)")
  p.add_argument("--lat_l2", type=float, default=1e-3)
  p.add_argument("--n_ref", type=int, default=1, help="references wm_eval.Eval retrieves (ignored by the field)")
  p.add_argument("--eval_n", type=int, default=300)
  p.add_argument("--eval_per_bin", type=int, default=6)
  p.add_argument("--eval_every", type=int, default=5000)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  prt.T_SCALE = np.ones(3)
  train = TokenSplit(os.path.join(D, "speedplus_tokens"), "train")
  val = {s: TokenSplit(os.path.join(D, "speedplus_tokens"), s) for s in ["val", "val_aug"]}
  meta = {}
  for s, d in [("train", train)] + list(val.items()):
    k, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", s))
    assert (idx == d.index).all() and np.abs(ts - d.t).max() < 1e-5, s
    meta[s] = k.astype(np.float32)
  bank = Bank(train, np.arange(len(train.R)), {int(train.obj[0]): np.eye(3)[None]}, 0.0)
  d_app = app_distance(D)
  print(f"appearance distance (val): {d_app:.4f}", flush=True)
  field = FeatureField(plane_res=a.plane_res, plane_dim=a.plane_dim, hid=a.hid, radius=a.radius, n_samp=a.n_samp, lat_dim=a.lat_dim,
                       geo=SpeedPlusGeometry())
  model = FieldModel(field=field)
  k1 = jnp.asarray(meta["train"][:1])
  params = model.init(jax.random.PRNGKey(a.seed), None, None, None, None, jnp.eye(3)[None], jnp.zeros((1, 3)), k1, jnp.zeros((1, 768)))
  print(f"params: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.2f}M", flush=True)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 1000, a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
  opt_state = tx.init(params)
  rays = (field.sub ** 2, a.n_samp)
  # per-training-image latents (lat_dim > 0), own optimiser; only the field's parameters are saved / evaluated
  lats = jnp.zeros((len(train.R), max(a.lat_dim, 1)))
  ltx = optax.adam(1e-2)
  lat_state = ltx.init(lats)

  @jax.jit
  def step(params, opt_state, lats, lat_state, key, b, R, t, k, tgt):
    def loss_fn(params, lats):
      jit_ = jax.random.uniform(key, (R.shape[0],) + rays)
      lat = lats[b] if a.lat_dim else None
      pred = model.apply(params, None, None, None, None, R, t, k, app_stats(tgt), jit_, lat=lat)
      reg = a.lat_l2 * (lats[b] ** 2).sum(-1).mean() if a.lat_dim else 0.0
      return ((pred - tgt) ** 2).mean() + reg, ((pred - tgt) ** 2).mean()

    (_, loss), (g, gl) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(params, lats)
    upd, opt_state = tx.update(g, opt_state, params)
    lu, lat_state = ltx.update(gl, lat_state, lats)
    return optax.apply_updates(params, upd), opt_state, optax.apply_updates(lats, lu), lat_state, loss

  evals = {s: Eval(d, meta[s], bank, a, 7) for s, d in val.items()}
  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  best = -np.inf

  def run_eval(step_i):
    nonlocal best
    t0 = time.time()
    rec = {"step": step_i}
    for s, ev in evals.items():
      rec[s] = ev.summary(ev.energies(model, params, train, meta["train"]), d_app)
    rec["eval_sec"] = time.time() - t0
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    print(f"[eval {step_i}] {json.dumps(rec)}", flush=True)
    if rec["val"]["P 1-3"] > best:
      best = rec["val"]["P 1-3"]
      open(os.path.join(a.out, "params.msgpack"), "wb").write(flax.serialization.to_bytes(params))

  rng = np.random.default_rng(a.seed)
  key = jax.random.PRNGKey(a.seed + 1)
  run_eval(0)
  t0 = time.time()
  for i in range(1, a.steps + 1):
    b = rng.integers(0, len(train.R), a.batch_size)
    key, k = jax.random.split(key)
    params, opt_state, lats, lat_state, loss = step(params, opt_state, lats, lat_state, k, jnp.asarray(b), jnp.asarray(train.R[b], jnp.float32),
                                                    jnp.asarray(train.t[b], jnp.float32), jnp.asarray(meta["train"][b]),
                                                    jnp.asarray(train.tokens[b], jnp.float32))
    if i % 100 == 0:
      rec = {"step": i, "loss": float(loss), "it_per_s": 100 / (time.time() - t0)}
      t0 = time.time()
      logf.write(json.dumps(rec) + "\n")
      logf.flush()
      if i % 1000 == 0:
        print(json.dumps(rec), flush=True)
    if i % a.eval_every == 0:
      run_eval(i)
  print("FIELD_DONE", flush=True)


if __name__ == "__main__":
  main()
