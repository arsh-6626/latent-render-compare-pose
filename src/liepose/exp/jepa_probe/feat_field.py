"""Feature field in a learned 3D-consistent feature space (SPEED+): an attempt at the render bottleneck.

Why (scripts/speedplus/roll_oracle.py, val, 600 held-out crops, the same comparator): with a perfect render (the
crop itself rotated, re-encoded) comparison in frozen DINOv2 space refines rolls to 0.07 deg (8x8) / 0.05 (16x16)
median; with the feature field's render it stops at 0.37 / 0.35 deg. Part of every DINOv2 token is not a property
of 3D points (context, the outline against the background, lighting) and no field renders it.

Here a head H maps each frozen DINOv2 token to --feat_dim features and the field (field.FeatureField, out_dim =
--feat_dim) renders H's space; both trained together on the training crops. Design (v2; v1 collapsed to a constant H,
then kept a per-image signature that the field copied through its appearance code, the contrast at chance):
  geometry only   --app none: the field gets no appearance input, so it renders what the pose determines and H must
                  map the token to what geometry alone predicts (3D-consistent by construction); --app stats as field.py
  H               f = LayerNorm_noaffine(W (z - mu) + MLP(z)): W, mu initialised to the whitened top --feat_dim PCA of
                  the training tokens (pose information from the start), the MLP's output layer zero-initialised
  reconstruction  |field(g*) - f|^2 at the GT pose / H's within-image per-channel variance          (--w_rec)
  contrast        softmax over -E / --tau (fixed) at g* and rotations of g*: --n_hard at 0.25-3 deg, --n_mid at
                  3-10, --n_easy at 10-30 (random axes) and the three 180-degree twins; target g*
  variance floor  relu(1 - within-image std of f) per channel (VICReg)                               (--w_var)
Logged every --eval_every steps on held-out val crops: pairwise P(E_pert > E_gt) at 1-3 / 3-10 deg in f-space (the
interpretable progress metric; training accuracy over mixed negatives is not). Gate, fixed before training: the roll
test in f-space (roll_oracle.py --ffield) <= 0.15 deg median after 4 steps (raw-space field 0.37, oracle 0.07).
A caveat to report with any result: H may also learn a pose code (a learned pose regressor rendered back), which is
still latent render-and-compare but no longer "frozen features".

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit python -m liepose.exp.jepa_probe.feat_field --out $LOGS/probe/sp_ffield_v2
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
from liepose.exp.jepa_probe.probe_frozen import perturb
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import SpeedPlusGeometry, crop_meta
from liepose.exp.jepa_probe.wm_eval import TWIN, app_stats


class Head(nn.Module):
  feat_dim: int = 64

  @nn.compact
  def __call__(self, z):
    lin = nn.Dense(self.feat_dim, name="pca")(z)
    res = nn.Dense(self.feat_dim, kernel_init=nn.initializers.zeros, name="res_out")(nn.gelu(nn.Dense(256, name="res_h")(nn.LayerNorm()(z))))
    return nn.LayerNorm(use_scale=False, use_bias=False)(lin + res)


class FeatField(nn.Module):
  feat_dim: int = 64
  plane_res: int = 128
  plane_dim: int = 32
  hid: int = 128
  n_samp: int = 32
  radius: float = 1.0
  use_app: bool = False

  def setup(self):
    self.field = FeatureField(plane_res=self.plane_res, plane_dim=self.plane_dim, hid=self.hid, out_dim=self.feat_dim,
                              radius=self.radius, n_samp=self.n_samp, geo=SpeedPlusGeometry())
    self.head = Head(self.feat_dim)

  def render(self, R, t, k, z, jitter=None, with_opacity=False):
    app = app_stats(z) if self.use_app else jnp.zeros((R.shape[0], 2 * z.shape[-1]))
    return self.field(R, t, k, app, jitter, with_opacity)

  def encode(self, z):
    return self.head(z)

  def __call__(self, R, t, k, z):
    return self.render(R, t, k, z), self.encode(z)


def build(a):
  return FeatField(feat_dim=a.feat_dim, plane_res=a.plane_res, plane_dim=a.plane_dim, hid=a.hid, n_samp=a.n_samp, radius=a.radius,
                   use_app=a.app == "stats")


def init_params(model, k):
  return model.init(jax.random.PRNGKey(0), jnp.eye(3)[None], jnp.zeros((1, 3)), k, jnp.zeros((1, 64, 384)))


def load(ff_dir, k):
  """(model, params) of a trained feat_field run"""
  a = argparse.Namespace(**json.load(open(os.path.join(ff_dir, "args.json"))))
  model = build(a)
  p0 = init_params(model, k)
  return model, jax.tree_util.tree_map(jnp.asarray, flax.serialization.from_bytes(p0, open(os.path.join(ff_dir, "params.msgpack"), "rb").read()))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--steps", type=int, default=30000)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--lr", type=float, default=2e-3)
  p.add_argument("--feat_dim", type=int, default=64)
  p.add_argument("--app", choices=["none", "stats"], default="none")
  p.add_argument("--n_hard", type=int, default=4)
  p.add_argument("--n_mid", type=int, default=3)
  p.add_argument("--n_easy", type=int, default=2)
  p.add_argument("--tau", type=float, default=0.05)
  p.add_argument("--w_rec", type=float, default=1.0)
  p.add_argument("--w_var", type=float, default=1.0)
  p.add_argument("--plane_res", type=int, default=128)
  p.add_argument("--plane_dim", type=int, default=32)
  p.add_argument("--hid", type=int, default=128)
  p.add_argument("--n_samp", type=int, default=32)
  p.add_argument("--radius", type=float, default=1.0)
  p.add_argument("--eval_n", type=int, default=200)
  p.add_argument("--eval_every", type=int, default=5000)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  train = TokenSplit(os.path.join(D, "speedplus_tokens"), "train")
  meta, idx, ts = crop_meta(os.path.join(D, "speedplus_crops", "train"))
  assert (idx == train.index).all() and np.abs(ts - train.t).max() < 1e-5
  val = TokenSplit(os.path.join(D, "speedplus_tokens"), "val")
  vmeta, vidx, _ = crop_meta(os.path.join(D, "speedplus_crops", "val"))
  assert (vidx == val.index).all()
  model = build(a)
  params = flax.core.unfreeze(init_params(model, jnp.asarray(meta[:1])))
  # H's linear part: whitened top-feat_dim PCA of training tokens
  zs = np.asarray(train.tokens[np.random.default_rng(0).choice(len(train.R), 2000, replace=False)], np.float64).reshape(-1, 384)
  mu = zs.mean(0)
  ev, U = np.linalg.eigh(np.cov((zs - mu).T))
  o = np.argsort(ev)[::-1][: a.feat_dim]
  W = U[:, o] / np.sqrt(ev[o] + 1e-6)
  params["params"]["head"]["pca"] = {"kernel": jnp.asarray(W, jnp.float32), "bias": jnp.asarray(-mu @ W, jnp.float32)}
  print(f"params: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.2f}M; PCA {a.feat_dim} dims keep "
        f"{ev[o].sum() / ev.sum():.2f} of the token variance", flush=True)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 1000, a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
  opt_state = tx.init(params)
  C = 1 + a.n_hard + a.n_mid + a.n_easy + 3
  twin = jnp.asarray(TWIN, jnp.float32)

  def rots(key, R, n, lo, hi):
    N = R.shape[0]
    k1, k2 = jax.random.split(key)
    ax = jax.random.normal(k1, (N, n, 3))
    ax = ax / jnp.linalg.norm(ax, axis=-1, keepdims=True)
    ang = jnp.radians(jnp.exp(jax.random.uniform(k2, (N, n, 1), minval=jnp.log(lo), maxval=jnp.log(hi))))
    return expm((ax * ang).reshape(-1, 3)).reshape(N, n, 3, 3) @ R[:, None]

  @jax.jit
  def step(params, opt_state, key, R, t, k, z):
    N = R.shape[0]
    kh, km, ke, kj = jax.random.split(key, 4)
    Rc = jnp.concatenate([R[:, None], rots(kh, R, a.n_hard, 0.25, 3.0), rots(km, R, a.n_mid, 3.0, 10.0), rots(ke, R, a.n_easy, 10.0, 30.0),
                          R[:, None] @ twin[None]], 1).reshape(N * C, 3, 3)
    rep = lambda x: jnp.repeat(x, C, 0)

    def loss_fn(params):
      f = model.apply(params, z, method=FeatField.encode)
      jit_ = jax.random.uniform(kj, (N * C, 256, a.n_samp))
      r = model.apply(params, Rc, rep(t), rep(k), rep(z), jit_, method=FeatField.render).reshape(N, C, 64, -1)
      var = f.var(1).mean(0)
      E = (((r - f[:, None]) ** 2) / (var + 1e-4)).mean((-1, -2))
      nce = optax.softmax_cross_entropy_with_integer_labels(-E / a.tau, jnp.zeros(N, jnp.int32)).mean()
      rec = E[:, 0].mean()
      vfloor = jax.nn.relu(1.0 - jnp.sqrt(var + 1e-4)).mean()
      return nce + a.w_rec * rec + a.w_var * vfloor, (nce, rec, vfloor)

    (loss, (nce, rec, vf)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss, nce, rec, vf

  # held-out val: pairwise P in f-space
  erng = np.random.default_rng(7)
  er = np.sort(erng.choice(len(val.R), a.eval_n, replace=False))
  ER = np.concatenate([val.R[er][:, None], perturb(np.repeat(val.R[er], 6, 0), 3.0, 1.0, erng).reshape(-1, 6, 3, 3),
                       perturb(np.repeat(val.R[er], 6, 0), 10.0, 3.0, erng).reshape(-1, 6, 3, 3)], 1)

  @jax.jit
  def val_E(params, R, t, k, z):
    f = model.apply(params, z, method=FeatField.encode)
    r = model.apply(params, R.reshape(-1, 3, 3), jnp.repeat(t, 13, 0), jnp.repeat(k, 13, 0), jnp.repeat(z, 13, 0), method=FeatField.render)
    r = r.reshape(z.shape[0], 13, 64, -1)
    return (((r - f[:, None]) ** 2) / (f.var(1).mean(0) + 1e-4)).mean((-1, -2))

  def evaluate(params):
    E = np.concatenate([np.asarray(val_E(params, jnp.asarray(ER[s:s + 8], jnp.float32), jnp.asarray(val.t[er[s:s + 8]], jnp.float32),
                                         jnp.asarray(vmeta[er[s:s + 8]]), jnp.asarray(val.tokens[er[s:s + 8]], jnp.float32)))
                        for s in range(0, a.eval_n, 8)])
    return {"P 1-3": float((E[:, 1:7] > E[:, :1]).mean()), "P 3-10": float((E[:, 7:13] > E[:, :1]).mean())}

  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  ev0 = evaluate(params)
  print(f"[eval 0] {json.dumps(ev0)}", flush=True)
  logf.write(json.dumps({"step": 0, "val": ev0}) + "\n")
  rng = np.random.default_rng(a.seed)
  key = jax.random.PRNGKey(a.seed + 1)
  t0 = time.time()
  for i in range(1, a.steps + 1):
    b = rng.integers(0, len(train.R), a.batch_size)
    key, kk = jax.random.split(key)
    params, opt_state, loss, nce, rec, vf = step(params, opt_state, kk, jnp.asarray(train.R[b], jnp.float32), jnp.asarray(train.t[b], jnp.float32),
                                                 jnp.asarray(meta[b]), jnp.asarray(train.tokens[b], jnp.float32))
    if i % 100 == 0:
      r_ = {"step": i, "loss": float(loss), "nce": float(nce), "rec": float(rec), "vfloor": float(vf), "it_per_s": 100 / (time.time() - t0)}
      t0 = time.time()
      logf.write(json.dumps(r_) + "\n")
      logf.flush()
    if i % a.eval_every == 0 or i == a.steps:
      ev = evaluate(params)
      logf.write(json.dumps({"step": i, "val": ev}) + "\n")
      logf.flush()
      print(f"[eval {i}] {json.dumps(ev)} nce {float(nce):.3f} rec {float(rec):.3f}", flush=True)
      open(os.path.join(a.out, "params.msgpack"), "wb").write(flax.serialization.to_bytes(params))
  print("FEAT_FIELD_DONE", flush=True)


if __name__ == "__main__":
  main()
