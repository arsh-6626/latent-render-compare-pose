"""Probe v3: pose-conditioned predictor on a frozen encoder (cached DINOv2 ViT-S/14 patch tokens).

Only the predictor trains, as in V-JEPA 2-AC. Context is always the canonical reference
(identity pose), so ẑ(g) = P(z_ref, g) is exactly the call the diffusion loop will make.

Loss per observation: the GT pose competes with K uniform-random and K near-GT rotations
(same translation) under a foreground-weighted latent energy; negatives are never filtered by
symmetry. A weighted regression term on the GT prediction keeps ẑ close to the target.

  python -m liepose.exp.jepa_probe.probe_frozen --cache $DATADIR/dino_cache --out $DATADIR/logs/probe/v3
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.spatial.transform import Rotation

from liepose.data.symsolt.dataset import SHAPE_NAMES
from liepose.data.symsolt.presets import preset_dict
from liepose.exp.jepa_probe.probe import PosePredictor, geodesic_deg, pose_feat, wxyz_to_mat

FG_FLOOR = 0.05  # weight kept on background patches


# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

def pool_tokens(tok, fg, pool):
  """(N,256,D) tokens on a 16x16 grid -> (N,(16/pool)^2,D); fg (N,16,16) -> (N,(16/pool)^2)."""
  if pool == 1:
    return tok, fg.reshape(len(fg), -1)
  n, _, d = tok.shape
  g = 16 // pool
  tok = tok.reshape(n, g, pool, g, pool, d).mean(axis=(2, 4), dtype=np.float32).astype(np.float16)
  fg = fg.reshape(n, g, pool, g, pool).max(axis=(2, 4))
  return tok.reshape(n, g * g, d), fg.reshape(n, g * g)


class CachedSplit:
  def __init__(self, cache, split, pool, shapes=None, max_per_shape=None):
    meta_path = os.path.join(cache, f"{split}_meta.npz")
    if not os.path.isfile(meta_path):
      meta_path = os.path.join(cache, f"{split}_p{pool}_meta.npz")
    meta = np.load(meta_path)
    label = meta["label"]
    keep = np.ones(len(label), bool)
    if shapes is not None:
      keep &= np.isin(label, [SHAPE_NAMES.index(s) for s in shapes])
    idx = np.nonzero(keep)[0]
    if max_per_shape is not None:
      idx = np.concatenate([idx[label[idx] == s][:max_per_shape] for s in np.unique(label[idx])])
    t0 = time.time()
    pre = os.path.join(cache, f"{split}_p{pool}_tokens.npy")
    if pool > 1 and os.path.isfile(pre):  # cache already stored 2x2-pooled
      self.tokens = np.ascontiguousarray(np.load(pre, mmap_mode="r")[idx])
      self.fg = np.load(os.path.join(cache, f"{split}_p{pool}_fg.npy"))[idx].astype(np.float32)
    else:
      tok = np.load(os.path.join(cache, f"{split}_tokens.npy"), mmap_mode="r")
      fg = np.load(os.path.join(cache, f"{split}_fg.npy"), mmap_mode="r")
      parts = [pool_tokens(np.asarray(tok[idx[i : i + 4096]]), np.asarray(fg[idx[i : i + 4096]]), pool)
               for i in range(0, len(idx), 4096)]
      self.tokens = np.concatenate([p[0] for p in parts])
      self.fg = np.concatenate([p[1] for p in parts]).astype(np.float32)
    print(f"{split}: {len(idx)} samples, tokens {self.tokens.shape} in {time.time() - t0:.0f}s", flush=True)
    self.R = wxyz_to_mat(meta["rotation"][idx])
    self.t = meta["translation"][idx]
    self.label = label[idx]
    self.by_shape = {s: np.nonzero(self.label == s)[0] for s in np.unique(self.label)}

  def equivalents(self, i):
    pose = np.eye(4)
    pose[:3, :3] = self.R[i]
    return preset_dict[SHAPE_NAMES[self.label[i]]].get_symmetric_poses(pose)[:, :3, :3]


class PairSet:
  """Rendered GT + fine-perturbation candidates of training poses (scripts/oracle/pair_render.py, pair_encode.py),
  each with its own token target: supervision at p + delta, which the observed images alone never give."""

  def __init__(self, path, pool):
    t0 = time.time()
    meta = np.load(os.path.join(path, "pairs_meta.npz"))
    tok = np.load(os.path.join(path, "pairs_tokens.npy"), mmap_mode="r")
    fg = np.load(os.path.join(path, "pairs_fg.npy"), mmap_mode="r")
    M, C = tok.shape[:2]
    if tok.shape[2] == (16 // pool) ** 2 and pool > 1:  # stored already pooled (roll_encode.py)
      self.tokens, self.fg = np.asarray(tok), np.asarray(fg, np.float32)
    else:
      parts = [pool_tokens(np.asarray(tok[i : i + 256]).reshape(-1, 256, tok.shape[-1]),
                           np.asarray(fg[i : i + 256]).reshape(-1, 16, 16), pool) for i in range(0, M, 256)]
      self.tokens = np.concatenate([p[0] for p in parts]).reshape(M, C, -1, tok.shape[-1])
      self.fg = np.concatenate([p[1] for p in parts]).reshape(M, C, -1).astype(np.float32)
    self.g = pose_feat(meta["R"], meta["t"])
    self.R, self.t, self.index = meta["R"], meta["t"], meta["index"]
    self.rot_off = meta["rot_off"]
    self.label = meta["label"]
    print(f"pairs: {M} poses x {C} candidates, tokens {self.tokens.shape} in {time.time() - t0:.0f}s", flush=True)


def sample_pairs(rng, pairs, batch_size):
  b = rng.integers(0, len(pairs.label), batch_size)
  return {"p_shape": pairs.label[b], "p_g": pairs.g[b], "p_tgt": pairs.tokens[b], "p_fg": pairs.fg[b]}


def perturb(R, max_deg, min_deg, rng):
  """Rotate each R by a random axis with angle in [min_deg, max_deg]."""
  n = R.shape[0]
  axis = rng.normal(size=(n, 3))
  axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
  ang = np.radians(rng.uniform(min_deg, max_deg, size=(n, 1)))
  return Rotation.from_rotvec(axis * ang).as_matrix() @ R


def sample_batch(rng, data, batch_size, n_uni, n_near, near_min_deg=5.0, near_max_deg=30.0,
                 n_tuni=0, n_tnear=0, tnear_std=0.1):
  """Candidates: [GT | n_uni random R | n_near near R | n_tuni random t | n_tnear near t].
  Rotation candidates keep the GT translation and translation candidates keep the GT rotation."""
  b = rng.integers(0, len(data.label), batch_size)
  R = np.repeat(data.R[b][:, None], 1 + n_uni + n_near + n_tuni + n_tnear, axis=1)
  if n_uni:
    R[:, 1 : 1 + n_uni] = Rotation.random(batch_size * n_uni, random_state=rng.integers(2**31)).as_matrix().reshape(
      batch_size, n_uni, 3, 3)
  if n_near:
    R[:, 1 + n_uni : 1 + n_uni + n_near] = perturb(R[:, 1 + n_uni : 1 + n_uni + n_near].reshape(-1, 3, 3), near_max_deg, near_min_deg, rng).reshape(batch_size, n_near, 3, 3)
  t = np.repeat(data.t[b][:, None], R.shape[1], axis=1)
  o = 1 + n_uni + n_near
  if n_tuni:
    t[:, o : o + n_tuni] = rng.uniform(-1, 1, size=(batch_size, n_tuni, 3))
  if n_tnear:
    t[:, o + n_tuni :] += rng.normal(0, tnear_std, size=(batch_size, n_tnear, 3))
  # distance of every candidate to the GT pose (for soft targets)
  tr = np.einsum("bcij,bij->bc", R, data.R[b])
  d_gt = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).astype(np.float32)
  dt_gt = np.linalg.norm(t - data.t[b][:, None], axis=-1).astype(np.float32)
  return {"shape": data.label[b], "tgt": data.tokens[b], "fg": data.fg[b], "g_cand": pose_feat(R, t), "d_gt": d_gt,
          "dt_gt": dt_gt, "idx": b}


# ------------------------------------------------------------------------------
# Model / loss
# ------------------------------------------------------------------------------

class FrozenPoseModel(nn.Module):
  dim: int = 384
  depth: int = 6
  heads: int = 6
  out_dim: int = 384

  def setup(self):
    self.predictor = PosePredictor(dim=self.dim, depth=self.depth, heads=self.heads)
    self.proj = nn.Dense(self.out_dim)
    self.log_tau = self.param("log_tau", nn.initializers.constant(-2.0), ())

  def __call__(self, z_ref, g):
    return self.proj(self.predictor(z_ref, g))


def weighted_energy(pred, tgt, w):
  """pred (...,P,D), tgt (...,P,D), w (...,P) -> (...)"""
  return (w * ((pred - tgt) ** 2).mean(-1)).sum(-1) / w.sum(-1)


def make_train_step(model, tx, refs, w_rec, soft_sigma, sigma_t=0.0, w_pair=0.0):
  refs = jnp.asarray(refs, jnp.float32)

  def loss_fn(params, batch):
    g = batch["g_cand"]
    B, C = g.shape[:2]
    z_ref = jnp.repeat(refs[batch["shape"]], C, axis=0)
    pred = model.apply(params, z_ref, g.reshape(B * C, -1)).reshape(B, C, *refs.shape[1:])
    tgt = batch["tgt"].astype(jnp.float32)
    w = FG_FLOOR + batch["fg"]
    e = weighted_energy(pred, tgt[:, None], w[:, None])  # (B, C)
    tau = jnp.exp(params["params"]["log_tau"])
    if soft_sigma > 0:
      # smooth basin: credit decays with angular distance to the GT (no symmetry labels used)
      logit_t = -0.5 * (batch["d_gt"] / soft_sigma) ** 2
      if sigma_t > 0:
        logit_t = logit_t - 0.5 * (batch["dt_gt"] / sigma_t) ** 2
      target = jax.nn.softmax(logit_t, axis=-1)
      pose_nce = optax.softmax_cross_entropy(-e / tau, target).mean()
    else:
      pose_nce = optax.softmax_cross_entropy_with_integer_labels(-e / tau, jnp.zeros(B, jnp.int32)).mean()
    rec = e[:, 0].mean()
    loss = pose_nce + w_rec * rec
    logs = {"pose_nce": pose_nce, "rec": rec, "tau": tau, "pose_acc": (jnp.argmin(e, axis=1) == 0).mean()}
    if w_pair > 0:  # regression to the rendered target at every candidate pose
      gp = batch["p_g"]
      Bp, Cp = gp.shape[:2]
      zp = jnp.repeat(refs[batch["p_shape"]], Cp, axis=0)
      pp = model.apply(params, zp, gp.reshape(Bp * Cp, -1)).reshape(Bp, Cp, *refs.shape[1:])
      pair = weighted_energy(pp, batch["p_tgt"].astype(jnp.float32), FG_FLOOR + batch["p_fg"]).mean()
      loss = loss + w_pair * pair
      logs["pair"] = pair
    logs["loss"] = loss
    return loss, logs

  @jax.jit
  def step(params, opt_state, batch):
    grads, logs = jax.grad(loss_fn, has_aux=True)(params, batch)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, logs

  return step


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def make_energy_fn(model, refs, chunk):
  refs = jnp.asarray(refs, jnp.float32)

  @jax.jit
  def energies(params, shape, tgt, fg, g):
    w = FG_FLOOR + fg
    tgt = tgt.astype(jnp.float32)

    def one_chunk(gc):
      zr = jnp.broadcast_to(refs[shape], (gc.shape[0],) + refs.shape[1:])
      return weighted_energy(model.apply(params, zr, gc), tgt[None], w[None])

    return jax.lax.map(one_chunk, g.reshape(-1, chunk, g.shape[-1])).reshape(-1)

  return energies


def evaluate(energy_fn, params, data, n_per_shape, n_rand, seed=0):
  R_rand = Rotation.random(n_rand, random_state=seed).as_matrix()
  t_rand = np.random.default_rng(seed).uniform(-1, 1, size=(255, 3))
  out = {}
  for s in sorted(data.by_shape):
    rows = {k: [] for k in ["pct", "top1", "prec", "chance", "tpct", "ttop1"]}
    for i in data.by_shape[s][:n_per_shape]:
      R_all = np.concatenate([data.R[i][None], R_rand])
      pad = (-len(R_all)) % 256
      R_all = np.concatenate([R_all, R_rand[:pad]])
      g = pose_feat(R_all, np.broadcast_to(data.t[i], (len(R_all), 3)))
      e = np.asarray(energy_fn(params, int(s), data.tokens[i], data.fg[i], g))
      e_gt, e_rand = e[0], e[1 : 1 + n_rand]
      d_eq = geodesic_deg(R_rand, data.equivalents(i))
      order = np.argsort(e_rand)
      k = max(1, n_rand // 100)
      rows["pct"].append((e_rand < e_gt).mean() + 0.5 * (e_rand == e_gt).mean())
      rows["top1"].append(d_eq[order[0]])
      rows["prec"].append((d_eq[order[:k]] < 30).mean())
      rows["chance"].append((d_eq < 30).mean())
      # translation: GT vs 255 random translations at the GT rotation
      t_all = np.concatenate([data.t[i][None], t_rand])
      et = np.asarray(energy_fn(params, int(s), data.tokens[i], data.fg[i],
                                pose_feat(np.broadcast_to(data.R[i], (256, 3, 3)), t_all)))
      rows["tpct"].append((et[1:] < et[0]).mean() + 0.5 * (et[1:] == et[0]).mean())
      rows["ttop1"].append(np.linalg.norm(t_all[np.argmin(et)] - data.t[i]))
    pct = np.array(rows["pct"])
    out[SHAPE_NAMES[s]] = {
      "gt_percentile_mean": float(pct.mean()),
      "gt_in_top5pct": float((pct < 0.05).mean()),
      "top1_err_to_equiv_deg_median": float(np.median(rows["top1"])),
      "top1pct_within30deg_of_equiv": float(np.mean(rows["prec"])),
      "chance_within30": float(np.mean(rows["chance"])),
      "lift": float(np.mean(rows["prec"]) / max(np.mean(rows["chance"]), 1e-9)),
      "t_gt_percentile": float(np.mean(rows["tpct"])),
      "t_top1_err_median": float(np.median(rows["ttop1"])),
    }
  names = [k for k in out]
  out["mean"] = {k: float(np.mean([out[n][k] for n in names])) for k in out[names[0]]}
  return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--cache", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--pool", type=int, default=2, help="average-pool the 16x16 token grid by this factor")
  p.add_argument("--shapes", nargs="+", default=None)
  p.add_argument("--max_train_per_shape", type=int, default=None)
  p.add_argument("--steps", type=int, default=30000)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--n_uni", type=int, default=15)
  p.add_argument("--n_near", type=int, default=16)
  p.add_argument("--w_rec", type=float, default=1.0)
  p.add_argument("--soft_sigma", type=float, default=0.0, help="deg; >0 uses Gaussian soft targets over candidates")
  p.add_argument("--near_min_deg", type=float, default=5.0)
  p.add_argument("--near_max_deg", type=float, default=30.0)
  p.add_argument("--n_tuni", type=int, default=0, help="random-translation candidates (GT rotation)")
  p.add_argument("--n_tnear", type=int, default=0, help="near-translation candidates (GT rotation)")
  p.add_argument("--tnear_std", type=float, default=0.1)
  p.add_argument("--sigma_t", type=float, default=0.0, help=">0 adds translation distance to the soft targets")
  p.add_argument("--train_split", default="train-45k")
  p.add_argument("--pairs", default=None, help="dir with pairs_{tokens,fg,meta} (paired supervision at p + delta)")
  p.add_argument("--w_pair", type=float, default=1.0)
  p.add_argument("--pair_batch", type=int, default=8)
  p.add_argument("--init", default=None, help="probe dir whose params.msgpack initialises the predictor")
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--dim", type=int, default=384)
  p.add_argument("--depth", type=int, default=6)
  p.add_argument("--eval_every", type=int, default=2500)
  p.add_argument("--eval_per_shape", type=int, default=50)
  p.add_argument("--eval_rand", type=int, default=2048)
  p.add_argument("--eval_train", action="store_true", help="also evaluate on training samples (overfit check)")
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  with open(os.path.join(a.out, "args.json"), "w") as f:
    json.dump(vars(a), f, indent=2)
  print(jax.devices(), flush=True)

  train = CachedSplit(a.cache, a.train_split, a.pool, a.shapes, a.max_train_per_shape)
  test = CachedSplit(a.cache, "test-5k", a.pool, a.shapes, a.eval_per_shape)
  refs, _ = pool_tokens(np.load(os.path.join(a.cache, "refs_tokens.npy")),
                        np.load(os.path.join(a.cache, "refs_fg.npy")), a.pool)

  model = FrozenPoseModel(dim=a.dim, depth=a.depth, heads=a.dim // 64, out_dim=refs.shape[-1])
  params = model.init(jax.random.PRNGKey(a.seed), jnp.zeros((1,) + refs.shape[1:]), jnp.zeros((1, 9)))
  if a.init:
    with open(os.path.join(a.init, "params.msgpack"), "rb") as f:
      params = flax.serialization.from_bytes(params, f.read())
    print(f"initialised from {a.init}", flush=True)
  print(f"params: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.1f}M", flush=True)
  warmup = min(1000, a.steps // 10)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, warmup, a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.05))
  opt_state = tx.init(params)
  pairs = PairSet(a.pairs, a.pool) if a.pairs else None
  step_fn = make_train_step(model, tx, refs, a.w_rec, a.soft_sigma, a.sigma_t, a.w_pair if pairs else 0.0)
  energy_fn = make_energy_fn(model, refs, chunk=256)
  logf = open(os.path.join(a.out, "log.jsonl"), "a")

  def run_eval(step):
    t0 = time.time()
    rec = {"step": step, "eval": evaluate(energy_fn, params, test, a.eval_per_shape, a.eval_rand)}
    if a.eval_train:
      rec["eval_train"] = evaluate(energy_fn, params, train, a.eval_per_shape, a.eval_rand)
    rec["eval_sec"] = time.time() - t0
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    print(f"[eval {step}] test " + json.dumps(rec["eval"]["mean"]), flush=True)
    if a.eval_train:
      print(f"[eval {step}] train " + json.dumps(rec["eval_train"]["mean"]), flush=True)

  rng = np.random.default_rng(a.seed)
  run_eval(0)
  pool = ThreadPoolExecutor(1)

  def next_batch():
    b = sample_batch(rng, train, a.batch_size, a.n_uni, a.n_near, a.near_min_deg, a.near_max_deg,
                     a.n_tuni, a.n_tnear, a.tnear_std)
    if pairs is not None:
      b.update(sample_pairs(rng, pairs, a.pair_batch))
    return b

  fut = pool.submit(next_batch)
  t0 = time.time()
  for step in range(1, a.steps + 1):
    batch = fut.result()
    fut = pool.submit(next_batch)
    params, opt_state, logs = step_fn(params, opt_state, batch)
    if step % 100 == 0:
      logs = {k: float(v) for k, v in logs.items()}
      logs.update(step=step, it_per_s=100 / (time.time() - t0))
      t0 = time.time()
      logf.write(json.dumps(logs) + "\n")
      logf.flush()
      if step % 500 == 0:
        print(json.dumps(logs), flush=True)
    if step % a.eval_every == 0 or step == a.steps:
      run_eval(step)

  with open(os.path.join(a.out, "params.msgpack"), "wb") as f:
    f.write(flax.serialization.to_bytes(params))
  print("done", flush=True)


if __name__ == "__main__":
  main()
