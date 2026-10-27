"""Is the render or the token space the limit of fine comparison? (SPEED+, diagnostic; GT poses / exact warps used.)

One task, two renders, the same comparator (liepose.exp.jepa_probe.comparator.Comparator, output 0 = the roll in
deg), the same training and evaluation:
  task      the observation and a rendering offset by a roll theta ~ U(-10, 10) deg -> predict theta
  oracle    the rendering is the real crop exactly rotated in-plane by theta about the crop centre, encoded by frozen
            DINOv2 (flax port, 2x2-pooled as the caches); blur-matched: the observation is the same crop rotated by
            --base deg and the rendering by --base + theta, so resampling blur carries no information. A perfect
            render: the precision any comparator can reach in this token space.
  field     the rendering is the feature field (field.py, frozen) at the GT pose rolled by theta about the
            allocentric viewing axis; the observation is the crop's cached tokens (unwarped).
Training on the training crops, evaluation on held-out synthetic val crops: median |error| of one step and after
--iters re-rendered steps, per |theta| bin. If oracle is precise and field is not, the render is the limit; if both
fail at 1-3 deg, the token space (or the comparator) is.

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit python scripts/speedplus/roll_oracle.py --arm oracle --out <dir>
"""

import argparse
import glob
import json
import os
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from liepose.exp.jepa_probe.comparator import Comparator, load_field
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import crop_meta
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.models.dinov2 import DinoV2, pool2

BINS = [(0.25, 1.0), (1.0, 3.0), (3.0, 10.0)]


def rotate(img, deg):
  """img (H, W, 3) float, rotated in-plane by deg about the centre (bilinear, zero outside)"""
  H, W = img.shape[:2]
  th = jnp.radians(deg)
  yy, xx = jnp.meshgrid(jnp.arange(H) - (H - 1) / 2, jnp.arange(W) - (W - 1) / 2, indexing="ij")
  xs = jnp.cos(th) * xx + jnp.sin(th) * yy + (W - 1) / 2
  ys = -jnp.sin(th) * xx + jnp.cos(th) * yy + (H - 1) / 2
  return jnp.stack([jax.scipy.ndimage.map_coordinates(img[..., c], [ys, xs], order=1, mode="constant") for c in range(3)], -1)


def rz(deg):
  c, s = jnp.cos(jnp.radians(deg)), jnp.sin(jnp.radians(deg))
  z, o = jnp.zeros_like(c), jnp.ones_like(c)
  return jnp.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(-1, 3, 3)


def load_images(D, split, max_n):
  ims, idx = [], []
  for f in sorted(glob.glob(os.path.join(D, "speedplus_crops", split, "shard_*.npz"))):
    d = np.load(f)
    ims.append(d["image"])
    idx.append(d["index"])
    if sum(len(x) for x in ims) >= max_n:
      break
  return np.concatenate(ims)[:max_n], np.concatenate(idx)[:max_n]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--arm", choices=["oracle", "field"], required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--field", default=None, help="field dir (arm field)")
  p.add_argument("--steps", type=int, default=30000)
  p.add_argument("--batch_size", type=int, default=64)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--max_deg", type=float, default=10.0)
  p.add_argument("--base", type=float, default=30.0)
  p.add_argument("--n_train", type=int, default=40000, help="training crops held in memory (oracle arm)")
  p.add_argument("--n_eval", type=int, default=600)
  p.add_argument("--iters", type=int, default=4)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--grid", type=int, default=8, help="token grid (16: unpooled, with --tokens speedplus_tokens16 and a 16x16 field)")
  p.add_argument("--tokens", default="speedplus_tokens")
  p.add_argument("--proj", default=None, help="renderable_subspace.py npz: compare in the top --k S_e-whitened directions")
  p.add_argument("--k", type=int, default=16)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  cmp = Comparator()
  Dz = 384
  P = None
  if a.proj:
    P = jnp.asarray(np.load(a.proj)["V"][:, : a.k])  # (384, k)
    Dz = a.k
  meta_tr, idx_tr, _ = crop_meta(os.path.join(D, "speedplus_crops", "train"))
  meta_va, idx_va, _ = crop_meta(os.path.join(D, "speedplus_crops", "val"))
  T = a.grid ** 2
  params = cmp.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, T, Dz)), jnp.zeros((1, T, Dz)), jnp.zeros((1, T)), jnp.asarray(meta_tr[:1]))
  prj = (lambda z: z) if P is None else (lambda z: z @ P)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 1000, a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
  opt_state = tx.init(params)

  if a.arm == "oracle":
    dino = DinoV2()
    dp = flax.serialization.msgpack_restore(open(os.path.join(D, "dinov2_vits14_224.msgpack"), "rb").read())
    dp = jax.tree_util.tree_map(jnp.asarray, dp)
    im_tr, _ = load_images(D, "train", a.n_train)
    im_va, iv = load_images(D, "val", a.n_eval)
    assert (iv == idx_va[: len(iv)]).all()
    meta_tr, meta_va = meta_tr[: len(im_tr)], meta_va[: len(im_va)]
    enc = (lambda x: pool2(dino.apply(dp, x))) if a.grid == 8 else (lambda x: dino.apply(dp, x))
    rot = jax.vmap(rotate)

    def pair(x, theta, k):  # observation at base, rendering at base + theta
      z_obs = enc(rot(x, jnp.full(x.shape[0], a.base)))
      z_hat = enc(rot(x, a.base + theta))
      return z_obs, z_hat, jnp.ones(z_obs.shape[:2])

    def get(split, b):
      im, mt = (im_tr, meta_tr) if split == "train" else (im_va, meta_va)
      return jnp.asarray(im[b], jnp.float32), jnp.asarray(mt[b])
    n_tr, n_va = len(im_tr), len(im_va)
  else:
    fm, fp = load_field(a.field)
    tr = TokenSplit(os.path.join(D, a.tokens), "train")
    va = TokenSplit(os.path.join(D, a.tokens), "val")

    def pair(x, theta, k):  # x = (tokens, R, t): the field at the GT pose rolled by theta
      z, R, t = x
      z_hat, opa = fm.apply(fp, None, None, None, None, rz(theta) @ R, t, k, app_stats(z), with_opacity=True)
      return z, z_hat, opa

    def get(split, b):
      d, mt = (tr, meta_tr) if split == "train" else (va, meta_va)
      return (jnp.asarray(d.tokens[b], jnp.float32), jnp.asarray(d.R[b], jnp.float32), jnp.asarray(d.t[b], jnp.float32)), jnp.asarray(mt[b])
    n_tr, n_va = len(tr.R), min(a.n_eval, len(va.R))

  @jax.jit
  def step(params, opt_state, key, x, k):
    theta = jax.random.uniform(key, (k.shape[0],), minval=-a.max_deg, maxval=a.max_deg)
    z_obs, z_hat, opa = jax.lax.stop_gradient(pair(x, theta, k))
    z_obs, z_hat = prj(z_obs), prj(z_hat)

    def loss_fn(params):
      out = cmp.apply(params, z_obs, z_hat, opa, k)[:, 0]
      return optax.huber_loss(out, -theta, delta=1.0).mean(), jnp.abs(out + theta).mean()

    (loss, err), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss, err

  @jax.jit
  def predict(params, x, theta, k):
    z_obs, z_hat, opa = pair(x, theta, k)
    return cmp.apply(params, prj(z_obs), prj(z_hat), opa, k)[:, 0]

  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  rng = np.random.default_rng(a.seed)
  key = jax.random.PRNGKey(a.seed + 1)
  t0 = time.time()
  for i in range(1, a.steps + 1):
    x, k = get("train", np.sort(rng.integers(0, n_tr, a.batch_size)))
    key, kk = jax.random.split(key)
    params, opt_state, loss, err = step(params, opt_state, kk, x, k)
    if i % 500 == 0:
      rec = {"step": i, "loss": float(loss), "abs_err_deg": float(err), "it_per_s": 500 / (time.time() - t0)}
      t0 = time.time()
      logf.write(json.dumps(rec) + "\n")
      logf.flush()
      if i % 5000 == 0:
        print(json.dumps(rec), flush=True)
  open(os.path.join(a.out, "params.msgpack"), "wb").write(flax.serialization.to_bytes(params))

  # held-out val: theta per bin, one step and --iters re-rendered steps (the rendering re-made at the corrected offset)
  res = {"arm": a.arm, "n_eval": int(n_va), "proj": a.proj, "k": a.k if a.proj else None}
  erng = np.random.default_rng(11)
  for lo, hi in BINS:
    th = erng.uniform(lo, hi, n_va) * erng.choice([-1, 1], n_va)
    one, fin = [], []
    for s in range(0, n_va, 50):
      b = np.arange(s, min(s + 50, n_va))
      x, k = get("val", b)
      cur = jnp.asarray(th[b], jnp.float32)
      for it in range(a.iters):
        cur = cur + predict(params, x, cur, k)  # remaining offset after applying the predicted correction
        if it == 0:
          one.append(np.abs(np.asarray(cur)))
      fin.append(np.abs(np.asarray(cur)))
    one, fin = np.concatenate(one), np.concatenate(fin)
    res[f"{lo:g}-{hi:g}"] = {"start_median": float(np.median(np.abs(th))), "one_step_median": float(np.median(one)),
                             f"{a.iters}_steps_median": float(np.median(fin)), f"{a.iters}_steps_gt0.5": float((fin > 0.5).mean())}
    print(f"{lo:g}-{hi:g}", json.dumps(res[f"{lo:g}-{hi:g}"]), flush=True)
  json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
