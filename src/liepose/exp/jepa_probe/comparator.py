"""Learned comparator: the fine stage of latent render-and-compare on SPEED+ (after diffusion + 16-sample consensus).

Why: the feature field (field.py) renders a pose well at 3-30 deg, but about 0.5x an appearance change of what
frozen DINOv2 tokens hold cannot be rendered from any 3D model (bigger field, background removed, fitted lighting:
no change), twice the token change of a 1 deg rotation. A fixed energy |render(g) - z_obs|^2 therefore has its
minimum ~1.4 deg from the truth, and minimising it (CEM / gradient, scripts/speedplus/field_refine.py) moves
consensus estimates from 0.47 to ~3 deg. A learned comparison instead sees the field's unrenderable part at
training time, in the same form as at test time, and can learn to ignore it:

  C(z_obs, field(g), field(g) - z_obs, opacity(g), crop)  ->  (w, dt):  g <- (exp(w) R, t + dt), --iters times
  training: a training crop at its GT pose g*, perturbed g = (exp(u) R*, t* + v), u a random axis at a log-uniform
  angle in [--min_deg, --max_deg], v ~ N(0, s^2) per axis with s log-uniform in [--t_min, --t_max] (translation_scaled
  units); the frozen field renders g (its appearance code from z_obs, as at test); target (w, dt) = (-u, -v) to
  first order (exp(-u) exp(u) R* = R*); Huber loss on w in degrees and dt / --t_unit. Training crops only.

Inference reads the crop's tokens and box meta and the frozen field; the GT pose is read by the scorer only.
Gate, fixed before training (scripts/speedplus/comparator_refine.py, SPEED+ synthetic val, full split):
  1 bias: from the GT pose, --iters refinement steps drift <= 0.3 deg (median) (the field energy: 1.4 deg);
  2 pipeline: refined from the 16-sample consensus, official ESA and median rotation error both below consensus.
  3 convergence: from GT perturbed by 1-3 deg, median error after refinement <= 0.5 deg (added before any full-model
    result: a comparator that outputs ~0 passes 1 and 2 by doing nothing).

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit python -m liepose.exp.jepa_probe.comparator \
      --field $LOGS/probe/sp_field_v1 --out $LOGS/probe/sp_cmp_v1
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from liepose.exp.jepa_probe.field import FeatureField, FieldModel
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import SpeedPlusGeometry, crop_feat, crop_meta
from liepose.exp.jepa_probe.wm_eval import app_stats


def expm(w):
  """(N, 3) rotation vectors -> (N, 3, 3), smooth at 0"""
  th = jnp.sqrt((w ** 2).sum(-1, keepdims=True) + 1e-12)[..., None]
  k = w / th[..., 0]
  z = jnp.zeros_like(k[:, 0])
  K = jnp.stack([z, -k[:, 2], k[:, 1], k[:, 2], z, -k[:, 0], -k[:, 1], k[:, 0], z], -1).reshape(-1, 3, 3)
  return jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * K @ K


class Block(nn.Module):
  dim: int
  heads: int

  @nn.compact
  def __call__(self, x):
    x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads)(nn.LayerNorm()(x), nn.LayerNorm()(x))
    h = nn.LayerNorm()(x)
    return x + nn.Dense(self.dim)(nn.gelu(nn.Dense(4 * self.dim)(h)))


class Comparator(nn.Module):
  dim: int = 256
  depth: int = 6
  heads: int = 8

  @nn.compact
  def __call__(self, z_obs, z_hat, opa, k):
    """(N, 64, D) observed / rendered tokens, (N, 64) rendered opacity, (N, 6) crop meta -> (N, 6) = (w [deg], dt / t_unit)"""
    x = jnp.concatenate([nn.LayerNorm(name="ln_obs")(z_obs), nn.LayerNorm(name="ln_hat")(z_hat),
                         nn.LayerNorm(name="ln_diff")(z_hat - z_obs), opa[..., None]], -1)
    x = nn.Dense(self.dim, name="tok_in")(x) + self.param("pos", nn.initializers.normal(0.02), (1, x.shape[1], self.dim))
    c = nn.Dense(self.dim, name="crop_in")(crop_feat(k))[:, None] + self.param("cls", nn.initializers.normal(0.02), (1, 1, self.dim))
    x = jnp.concatenate([c, x], 1)
    for i in range(self.depth):
      x = Block(self.dim, self.heads, name=f"block_{i}")(x)
    return nn.Dense(6, kernel_init=nn.initializers.zeros, name="out")(nn.LayerNorm()(x[:, 0]))


def load_field(field_dir):
  fa = SimpleNamespace(**json.load(open(os.path.join(field_dir, "args.json"))))
  model = FieldModel(field=FeatureField(plane_res=fa.plane_res, plane_dim=fa.plane_dim, hid=getattr(fa, "hid", 128), radius=fa.radius,
                                        n_samp=fa.n_samp, lat_dim=getattr(fa, "lat_dim", 0), geo=SpeedPlusGeometry()))
  k0 = jnp.zeros((1, 6)).at[:, 0].set(1.0)
  p0 = model.init(jax.random.PRNGKey(0), None, None, None, None, jnp.eye(3)[None], jnp.zeros((1, 3)), k0, jnp.zeros((1, 768)))
  params = flax.serialization.from_bytes(p0, open(os.path.join(field_dir, "params.msgpack"), "rb").read())
  return model, jax.tree_util.tree_map(jnp.asarray, params)


def make_refine(cmp, field_model, field_params, iters, t_unit):
  """jitted (cmp_params, R (N,3,3), t (N,3), k (N,6), z (N,64,D)) -> refined R, t after `iters` steps"""

  @jax.jit
  def refine(cp, R, t, k, z):
    app = app_stats(z)
    for _ in range(iters):
      z_hat, opa = field_model.apply(field_params, None, None, None, None, R, t, k, app, with_opacity=True)
      out = cmp.apply(cp, z, z_hat, opa, k)
      R = expm(jnp.radians(out[:, :3])) @ R
      t = t + out[:, 3:] * t_unit
    return R, t

  return refine


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--field", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--steps", type=int, default=40000)
  p.add_argument("--batch_size", type=int, default=64)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--min_deg", type=float, default=0.05)
  p.add_argument("--max_deg", type=float, default=10.0)
  p.add_argument("--t_min", type=float, default=0.002)
  p.add_argument("--t_max", type=float, default=0.05)
  p.add_argument("--t_unit", type=float, default=0.01, help="translation_scaled units per output unit")
  p.add_argument("--dim", type=int, default=256)
  p.add_argument("--depth", type=int, default=6)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  train = TokenSplit(os.path.join(D, "speedplus_tokens"), "train")
  meta, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", "train"))
  assert (idx == train.index).all() and np.abs(ts - train.t).max() < 1e-5
  field_model, field_params = load_field(a.field)
  cmp = Comparator(dim=a.dim, depth=a.depth)
  params = cmp.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, 64, 384)), jnp.zeros((1, 64, 384)), jnp.zeros((1, 64)), jnp.asarray(meta[:1]))
  print(f"comparator params: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.2f}M", flush=True)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 1000, a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
  opt_state = tx.init(params)

  @jax.jit
  def step(params, opt_state, key, R, t, k, z):
    N = R.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)
    ax = jax.random.normal(k1, (N, 3))
    ax = ax / jnp.linalg.norm(ax, axis=-1, keepdims=True)
    ang = jnp.exp(jax.random.uniform(k2, (N, 1), minval=jnp.log(a.min_deg), maxval=jnp.log(a.max_deg)))
    u = ax * jnp.radians(ang)
    s = jnp.exp(jax.random.uniform(k3, (N, 1), minval=jnp.log(a.t_min), maxval=jnp.log(a.t_max)))
    v = jax.random.normal(k4, (N, 3)) * s
    Rp, tp = expm(u) @ R, t + v
    z_hat, opa = field_model.apply(field_params, None, None, None, None, Rp, tp, k, app_stats(z), with_opacity=True)
    z_hat, opa = jax.lax.stop_gradient(z_hat), jax.lax.stop_gradient(opa)
    tgt = jnp.concatenate([-jnp.degrees(u), -v / a.t_unit], -1)

    def loss_fn(params):
      out = cmp.apply(params, z, z_hat, opa, k)
      err = optax.huber_loss(out, tgt, delta=1.0)
      return err.mean(), (jnp.abs(out[:, :3] - tgt[:, :3]).mean(), jnp.abs(out[:, 3:] - tgt[:, 3:]).mean())

    (loss, (er, et)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss, er, et

  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  rng = np.random.default_rng(a.seed)
  key = jax.random.PRNGKey(a.seed + 1)
  t0 = time.time()
  for i in range(1, a.steps + 1):
    b = rng.integers(0, len(train.R), a.batch_size)
    key, kk = jax.random.split(key)
    params, opt_state, loss, er, et = step(params, opt_state, kk, jnp.asarray(train.R[b], jnp.float32), jnp.asarray(train.t[b], jnp.float32),
                                           jnp.asarray(meta[b]), jnp.asarray(train.tokens[b], jnp.float32))
    if i % 100 == 0:
      rec = {"step": i, "loss": float(loss), "rot_abs_err_deg": float(er), "t_abs_err_units": float(et), "it_per_s": 100 / (time.time() - t0)}
      t0 = time.time()
      logf.write(json.dumps(rec) + "\n")
      logf.flush()
      if i % 2000 == 0:
        print(json.dumps(rec), flush=True)
  open(os.path.join(a.out, "params.msgpack"), "wb").write(flax.serialization.to_bytes(params))
  print("COMPARATOR_DONE", flush=True)


if __name__ == "__main__":
  main()
