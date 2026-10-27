"""Feature field in a learned renderable feature space (SPEED+): the fix for the render bottleneck.

Why (scripts/speedplus/roll_oracle.py, val, 600 held-out crops, the same comparator): with a perfect render (the
crop itself rotated, re-encoded) comparison in frozen 8x8 DINOv2 space refines rolls to 0.07 deg (median, 4 steps);
with the feature field's render it stops at 0.37 deg. ~0.5x an appearance change of what DINOv2 tokens hold is not a
property of 3D points (context, the object's outline against the background) and no field renders it.

Here a head H maps each frozen DINOv2 token to --feat_dim features (per-token MLP, LayerNorm without affine, so the
features cannot shrink), and the field (field.FeatureField with out_dim = --feat_dim) renders H's space; both are
trained together on the training crops:
  reconstruction  |field(g*) - H(z)|^2 at the GT pose g*, divided by H's per-channel variance over each image's own
                  tokens (scale-free: a constant H gains nothing)                                  (weight --w_rec)
  variance floor  mean over channels of relu(1 - within-image std of H) (VICReg; within the image, else H keeps a
                  per-image signature the field copies through its appearance code)               (weight --w_var)
  pose contrast   softmax over the field's renders at g* and --n_neg rotations of g* at log-uniform 0.25-10 deg
                  (random axes), energies |field(g) - H(z)|^2 / tau, target g*                (weight 1)
The contrast keeps H pose-informative (a constant H cannot rank poses) while the reconstruction makes it renderable;
without the normalisation and the floor the first run collapsed (H constant, reconstruction 3e-4, contrast at chance).
Evaluation (scripts/speedplus/feat_field_eval.py): energy P(E_pert > E_gt) per bin, bias of gradient descent from
GT, and the roll comparator test in H's space against the same oracle. Gate, fixed before training (val): roll test
with the field render <= 0.15 deg (median, 4 steps; oracle in raw DINOv2 space 0.07, field in raw space 0.37).

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit python -m liepose.exp.jepa_probe.feat_field --out $LOGS/probe/sp_ffield_v1
"""

import argparse
import json
import os
import time

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from liepose.exp.jepa_probe.comparator import expm
from liepose.exp.jepa_probe.field import FeatureField
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import SpeedPlusGeometry, crop_meta
from liepose.exp.jepa_probe.wm_eval import app_stats


class Head(nn.Module):
  feat_dim: int = 64

  @nn.compact
  def __call__(self, z):
    h = nn.gelu(nn.Dense(256)(nn.LayerNorm()(z)))
    return nn.LayerNorm(use_scale=False, use_bias=False)(nn.Dense(self.feat_dim)(h))


class FeatField(nn.Module):
  """field + head; render(R, t, k, app) -> (N, 64, feat_dim), head(z) -> (N, 64, feat_dim)"""
  feat_dim: int = 64
  plane_res: int = 128
  plane_dim: int = 32
  hid: int = 128
  n_samp: int = 32
  radius: float = 1.0

  def setup(self):
    self.field = FeatureField(plane_res=self.plane_res, plane_dim=self.plane_dim, hid=self.hid, out_dim=self.feat_dim,
                              radius=self.radius, n_samp=self.n_samp, geo=SpeedPlusGeometry())
    self.head = Head(self.feat_dim)
    self.log_tau = self.param("log_tau", nn.initializers.constant(-1.0), ())

  def render(self, R, t, k, app, jitter=None, with_opacity=False):
    return self.field(R, t, k, app, jitter, with_opacity)

  def encode(self, z):
    return self.head(z)

  def __call__(self, R, t, k, z):
    return self.render(R, t, k, app_stats(z)), self.encode(z)


def build(a):
  return FeatField(feat_dim=a.feat_dim, plane_res=a.plane_res, plane_dim=a.plane_dim, hid=a.hid, n_samp=a.n_samp, radius=a.radius)


def init_params(model, k):
  return model.init(jax.random.PRNGKey(0), jnp.eye(3)[None], jnp.zeros((1, 3)), k, jnp.zeros((1, 64, 384)))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--steps", type=int, default=30000)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--lr", type=float, default=2e-3)
  p.add_argument("--feat_dim", type=int, default=64)
  p.add_argument("--n_neg", type=int, default=8)
  p.add_argument("--w_rec", type=float, default=1.0)
  p.add_argument("--w_var", type=float, default=1.0)
  p.add_argument("--plane_res", type=int, default=128)
  p.add_argument("--plane_dim", type=int, default=32)
  p.add_argument("--hid", type=int, default=128)
  p.add_argument("--n_samp", type=int, default=32)
  p.add_argument("--radius", type=float, default=1.0)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  train = TokenSplit(os.path.join(D, "speedplus_tokens"), "train")
  meta, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", "train"))
  assert (idx == train.index).all() and np.abs(ts - train.t).max() < 1e-5
  model = build(a)
  params = init_params(model, jnp.asarray(meta[:1]))
  print(f"params: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.2f}M", flush=True)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 1000, a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
  opt_state = tx.init(params)
  C = 1 + a.n_neg

  @jax.jit
  def step(params, opt_state, key, R, t, k, z):
    N = R.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    ax = jax.random.normal(k1, (N, a.n_neg, 3))
    ax = ax / jnp.linalg.norm(ax, axis=-1, keepdims=True)
    ang = jnp.radians(jnp.exp(jax.random.uniform(k2, (N, a.n_neg, 1), minval=jnp.log(0.25), maxval=jnp.log(10.0))))
    Rn = expm((ax * ang).reshape(-1, 3)).reshape(N, a.n_neg, 3, 3) @ R[:, None]
    Rc = jnp.concatenate([R[:, None], Rn], 1).reshape(N * C, 3, 3)
    rep = lambda x: jnp.repeat(x, C, 0)

    def loss_fn(params):
      f = model.apply(params, z, method=FeatField.encode)  # (N, 64, F)
      jit_ = jax.random.uniform(k3, (N * C, 256, a.n_samp))
      r = model.apply(params, Rc, rep(t), rep(k), rep(app_stats(z)), jit_, method=FeatField.render).reshape(N, C, 64, -1)
      var = f.var(1).mean(0)  # per channel, over each image's own tokens (spatial structure, not a global signature)
      E = (((r - f[:, None]) ** 2) / (var + 1e-4)).mean((-1, -2))  # (N, C), scale-free
      tau = jnp.exp(params["params"]["log_tau"])
      nce = optax.softmax_cross_entropy_with_integer_labels(-E / tau, jnp.zeros(N, jnp.int32)).mean()
      rec = E[:, 0].mean()
      vfloor = jax.nn.relu(1.0 - jnp.sqrt(var + 1e-4)).mean()
      return nce + a.w_rec * rec + a.w_var * vfloor, (nce, rec, (jnp.argmin(E, 1) == 0).mean(), tau)

    (loss, (nce, rec, acc, tau)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss, nce, rec, acc, tau

  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  rng = np.random.default_rng(a.seed)
  key = jax.random.PRNGKey(a.seed + 1)
  t0 = time.time()
  for i in range(1, a.steps + 1):
    b = rng.integers(0, len(train.R), a.batch_size)
    key, kk = jax.random.split(key)
    params, opt_state, loss, nce, rec, acc, tau = step(params, opt_state, kk, jnp.asarray(train.R[b], jnp.float32),
                                                       jnp.asarray(train.t[b], jnp.float32), jnp.asarray(meta[b]),
                                                       jnp.asarray(train.tokens[b], jnp.float32))
    if i % 100 == 0:
      rec_ = {"step": i, "loss": float(loss), "nce": float(nce), "rec": float(rec), "acc": float(acc), "tau": float(tau),
              "it_per_s": 100 / (time.time() - t0)}
      t0 = time.time()
      logf.write(json.dumps(rec_) + "\n")
      logf.flush()
      if i % 2000 == 0:
        print(json.dumps(rec_), flush=True)
    if i % 10000 == 0 or i == a.steps:
      open(os.path.join(a.out, "params.msgpack"), "wb").write(flax.serialization.to_bytes(params))
  print("FEAT_FIELD_DONE", flush=True)


if __name__ == "__main__":
  main()
