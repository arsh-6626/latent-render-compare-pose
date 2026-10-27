"""Day-7 probe: is a pose-conditioned JEPA latent energy pose-sensitive on SYMSOL-T?

Trains a context encoder E, an EMA target encoder E' and a pose-conditioned predictor P
on image pairs of the same shape, with the relative pose as the action:

  P(E(x_a), g_ab) ~ sg(E'(x_b)),   g_ab = g_b g_a^-1

With probability `p_ref` the context image is the canonical reference render (identity
pose), so g_ab = g_b. That is exactly the call the diffusion loop will make:
ẑ(g) = P(E(x_ref), g).

Evaluation scores the energy e(g) = mean((P(E(x_ref), g) - E'(x_obs))^2) on random
rotations plus the GT rotation and its symmetric equivalents (translation fixed at GT),
and reports whether the energy minima land on the equivalence set.

  python -m liepose.exp.jepa_probe.probe --data $DATADIR/symsolt --out $DATADIR/logs/probe/nce1 --w_nce 1.0
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import flax
import flax.linen as nn
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.spatial.transform import Rotation

from liepose.data.symsolt.dataset import SHAPE_NAMES
from liepose.data.utils import load_file
from liepose.models.djepa_jax import JEPAEncoder, JEPAPredictor, jepa_smooth_l1_loss

POSE_DIM = 9  # first two rotation columns (6) + translation (3)


# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

def wxyz_to_mat(q):
  q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
  return Rotation.from_quat(np.roll(q, -1, axis=-1)).as_matrix()


def pose_feat(R, t):
  """(…,3,3), (…,3) -> (…,9)"""
  cols = np.swapaxes(R[..., :, :2], -1, -2).reshape(*R.shape[:-2], 6)
  return np.concatenate([cols, t], axis=-1).astype(np.float32)


def relative_pose(Ra, ta, Rb, tb):
  Rab = Rb @ np.swapaxes(Ra, -1, -2)
  tab = tb - np.einsum("...ij,...j->...i", Rab, ta)
  return Rab, tab


class SymsoltSplit:
  """Whole split held in memory as uint8 (train-45k is ~6.8 GB)."""

  def __init__(self, path, color_mode="none", max_per_shape=None, with_equivalents=False):
    annos = load_file(os.path.join(path, "annotations.gz"))
    if max_per_shape is not None:
      kept, counts = [], {}
      for a in annos:
        c = counts.get(a["label_shape"], 0)
        if c < max_per_shape:
          kept.append(a)
          counts[a["label_shape"]] = c + 1
      annos = kept
    key = "image" if color_mode == "none" else f"image_{color_mode}"
    files = [os.path.join(path, a[key]) for a in annos]
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
      imgs = list(ex.map(lambda f: np.asarray(imageio.v2.imread(f))[..., :3], files))
    self.images = np.stack(imgs).astype(np.uint8)
    print(f"loaded {len(files)} images from {path} in {time.time() - t0:.0f}s")
    self.R = wxyz_to_mat(np.stack([a["rotation"] for a in annos]))
    self.t = np.stack([a["translation"] for a in annos]).astype(np.float64)
    self.label = np.array([a["label_shape"] for a in annos])
    self.by_shape = {s: np.nonzero(self.label == s)[0] for s in range(len(SHAPE_NAMES))}
    if with_equivalents:
      self.R_eq = [wxyz_to_mat(a["rotations_equivalent"]) for a in annos]


def load_refs(path, color_mode):
  refs_path = os.path.join(path, "refs.npz")
  assert os.path.isfile(refs_path), f"{refs_path} missing; run liepose.exp.jepa_probe.render_refs"
  return np.load(refs_path)[color_mode][..., :3].astype(np.uint8)


def sample_batch(rng, data, refs, batch_size, p_ref, n_neg=0):
  """Pairs of the same shape. g_cand[:, 0] is the true relative pose; g_cand[:, 1:] keep the
  true translation but swap in uniform-random rotations (pose-contrastive negatives, never
  filtered by symmetry, so equivalent poses are free to tie with the GT)."""
  b = rng.integers(0, len(data.label), batch_size)
  shapes = data.label[b]
  a = np.array([rng.choice(data.by_shape[s]) for s in shapes])
  use_ref = rng.random(batch_size) < p_ref

  Ra, ta = data.R[a].copy(), data.t[a].copy()
  Ra[use_ref] = np.eye(3)
  ta[use_ref] = 0.0
  Rb = np.repeat(data.R[b][:, None], 1 + n_neg, axis=1)
  if n_neg > 0:
    Rb[:, 1:] = Rotation.random(batch_size * n_neg, random_state=rng.integers(2**31)).as_matrix().reshape(
      batch_size, n_neg, 3, 3)
  tb = np.repeat(data.t[b][:, None], 1 + n_neg, axis=1)
  Rab, tab = relative_pose(Ra[:, None], ta[:, None], Rb, tb)

  xa = data.images[a].copy()
  xa[use_ref] = refs[shapes[use_ref]]
  return {"xa": xa, "xb": data.images[b], "g_cand": pose_feat(Rab, tab)}


def to_float(x):
  return x.astype(jnp.float32) / 127.5 - 1.0


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

def fourier_pose(g, n_freq=6):
  freqs = 2.0 ** jnp.arange(n_freq) * jnp.pi
  x = g[..., None] * freqs
  return jnp.concatenate([g, jnp.sin(x).reshape(*g.shape[:-1], -1), jnp.cos(x).reshape(*g.shape[:-1], -1)], -1)


class AdaLNBlock(nn.Module):
  """DiT-style block: the pose modulates every LayerNorm and gates every residual."""
  dim: int
  heads: int
  mlp_ratio: float = 4.0

  @nn.compact
  def __call__(self, x, c):
    mod = nn.Dense(6 * self.dim, kernel_init=nn.initializers.zeros)(nn.silu(c))[:, None]
    s1, b1, g1, s2, b2, g2 = jnp.split(mod, 6, axis=-1)
    ln = lambda y: nn.LayerNorm(use_scale=False, use_bias=False)(y)
    h = ln(x) * (1 + s1) + b1
    x = x + g1 * nn.MultiHeadDotProductAttention(num_heads=self.heads)(h, h)
    h = ln(x) * (1 + s2) + b2
    h = nn.Dense(self.dim)(nn.gelu(nn.Dense(int(self.dim * self.mlp_ratio))(h)))
    return x + g2 * h


class PosePredictor(nn.Module):
  """Pose enters as a prepended token and as adaLN modulation in every block."""
  dim: int
  depth: int
  heads: int

  @nn.compact
  def __call__(self, z, g):
    c = nn.Dense(self.dim)(nn.silu(nn.Dense(self.dim)(fourier_pose(g))))
    x = nn.Dense(self.dim, name="proj_in")(z)
    x = jnp.concatenate([c[:, None], x], axis=1)
    for i in range(self.depth):
      x = AdaLNBlock(self.dim, self.heads, name=f"block_{i}")(x, c)
    x = nn.LayerNorm()(x[:, 1:])
    return nn.Dense(self.dim, name="proj_out")(x)


class PoseWorldModel(nn.Module):
  dim: int = 384
  depth: int = 6
  pred_depth: int = 4
  heads: int = 6
  img_size: int = 224
  predictor_type: str = "additive"  # additive (D-JEPA predictor) | adaln

  def setup(self):
    self.encoder = JEPAEncoder(img_size=self.img_size, embed_dim=self.dim, depth=self.depth,
                               num_heads=self.heads, buffer_size=16)
    if self.predictor_type == "adaln":
      self.predictor = PosePredictor(dim=self.dim, depth=self.pred_depth, heads=self.heads)
    else:
      self.predictor = JEPAPredictor(embed_dim=self.dim, predictor_embed_dim=self.dim,
                                     action_dim=POSE_DIM, depth=self.pred_depth, num_heads=self.heads)
    self.log_tau = self.param("log_tau", nn.initializers.zeros, ())

  def encode(self, x):
    return self.encoder(to_float(x))

  def predict(self, z, g):
    return self.predictor(z, g)

  def __call__(self, xa, g, xb):
    return self.predict(self.encode(xa), g), self.encode(xb), self.log_tau


def pairwise_energy(pred, tgt):
  """Mean squared token distance between every prediction i and target j: (B, B)."""
  p = pred.reshape(pred.shape[0], -1)
  t = tgt.reshape(tgt.shape[0], -1)
  d = (p**2).sum(-1)[:, None] + (t**2).sum(-1)[None, :] - 2.0 * p @ t.T
  return d / p.shape[-1]


def make_train_step(model, tx, w_nce, w_pose, ema_tau):
  def loss_fn(params, ema_params, batch):
    g_cand = batch["g_cand"]  # (B, 1+K, 9)
    B, C = g_cand.shape[:2]
    z_a = model.apply(params, batch["xa"], method=PoseWorldModel.encode)
    z_rep = jnp.repeat(z_a, C, axis=0)
    pred_c = model.apply(params, z_rep, g_cand.reshape(B * C, -1), method=PoseWorldModel.predict)
    pred_c = pred_c.reshape(B, C, *pred_c.shape[1:])
    pred = pred_c[:, 0]
    log_tau = params["params"]["log_tau"]
    tgt = jax.lax.stop_gradient(model.apply(ema_params, batch["xb"], method=PoseWorldModel.encode))
    jepa = jepa_smooth_l1_loss(pred, tgt)
    d = pairwise_energy(pred, tgt)
    nce = optax.softmax_cross_entropy_with_integer_labels(-d / jnp.exp(log_tau), jnp.arange(B)).mean()
    acc = (jnp.argmin(d, axis=1) == jnp.arange(B)).mean()
    logs = {"jepa": jepa, "nce": nce, "nce_acc": acc}
    loss = jepa + w_nce * nce
    if C > 1:
      e = ((pred_c - tgt[:, None]) ** 2).mean(axis=(2, 3))  # (B, 1+K)
      pose_nce = optax.softmax_cross_entropy_with_integer_labels(-e / jnp.exp(log_tau), jnp.zeros(B, jnp.int32)).mean()
      logs.update(pose_nce=pose_nce, pose_acc=(jnp.argmin(e, axis=1) == 0).mean())
      loss = loss + w_pose * pose_nce
    logs.update(loss=loss, tau=jnp.exp(log_tau), tgt_std=jnp.std(tgt.mean(axis=1), axis=0).mean())
    return loss, logs

  @jax.jit
  def step(params, ema_params, opt_state, batch):
    grads, logs = jax.grad(loss_fn, has_aux=True)(params, ema_params, batch)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    ema_params = jax.tree_util.tree_map(lambda e, p: ema_tau * e + (1 - ema_tau) * p, ema_params, params)
    return params, ema_params, opt_state, logs

  return step


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def geodesic_deg(R, R_set):
  """min angle (deg) between each R (K,3,3) and a set (M,3,3) -> (K,)"""
  tr = np.einsum("kij,mij->km", R, R_set)
  ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
  return ang.min(axis=1)


def make_energy_fn(model, chunk):
  @jax.jit
  def energies(params, ema_params, x_ref, x_obs, g):
    z_ref = model.apply(params, x_ref[None], method=PoseWorldModel.encode)[0]
    z_obs = model.apply(ema_params, x_obs[None], method=PoseWorldModel.encode)[0]

    def one_chunk(gc):
      zr = jnp.broadcast_to(z_ref, (gc.shape[0],) + z_ref.shape)
      pred = model.apply(params, zr, gc, method=PoseWorldModel.predict)
      return ((pred - z_obs[None]) ** 2).mean(axis=(1, 2))

    return jax.lax.map(one_chunk, g.reshape(-1, chunk, POSE_DIM)).reshape(-1)

  return energies


def evaluate(energy_fn, params, ema_params, test, refs, n_per_shape, n_rand, n_eq, seed=0):
  rng = np.random.default_rng(seed)
  R_rand = Rotation.random(n_rand, random_state=seed).as_matrix()
  out = {}
  for s, name in enumerate(SHAPE_NAMES):
    idx = test.by_shape[s][:n_per_shape]
    pct, top1, prec, zscore = [], [], [], []
    for i in idx:
      R_eq = test.R_eq[i]
      if len(R_eq) > n_eq:
        R_eq = R_eq[rng.choice(len(R_eq), n_eq, replace=False)]
      R_all = np.concatenate([test.R[i][None], R_eq, R_rand], axis=0)
      pad = (-len(R_all)) % 256
      R_all = np.concatenate([R_all, R_rand[:pad]], axis=0)
      g = pose_feat(R_all, np.broadcast_to(test.t[i], (len(R_all), 3)))
      e = np.asarray(energy_fn(params, ema_params, refs[s], test.images[i], g))
      e_gt = e[0]
      e_rand = e[1 + len(R_eq): 1 + len(R_eq) + n_rand]
      pct.append((e_rand < e_gt).mean() + 0.5 * (e_rand == e_gt).mean())  # ties count half
      zscore.append((np.median(e_rand) - e_gt) / (e_rand.std() + 1e-12))
      order = np.argsort(e_rand)
      d_eq = geodesic_deg(R_rand[order[: max(1, n_rand // 100)]], test.R_eq[i])
      top1.append(d_eq[0])
      prec.append((d_eq < 30).mean())
    out[name] = {
      "gt_percentile_mean": float(np.mean(pct)),   # fraction of random rotations beating GT; lower is better
      "gt_in_top5pct": float(np.mean(np.array(pct) < 0.05)),
      "gt_zscore_median": float(np.median(zscore)),
      "top1_err_to_equiv_deg_median": float(np.median(top1)),
      "top1pct_within30deg_of_equiv": float(np.mean(prec)),
    }
  out["mean"] = {k: float(np.mean([out[n][k] for n in SHAPE_NAMES])) for k in out[SHAPE_NAMES[0]]}
  return out


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True, help="dir containing train-45k/ and test-5k/")
  p.add_argument("--out", required=True)
  p.add_argument("--color_mode", default="none")
  p.add_argument("--steps", type=int, default=20000)
  p.add_argument("--batch_size", type=int, default=64)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--w_nce", type=float, default=1.0)
  p.add_argument("--p_ref", type=float, default=0.5)
  p.add_argument("--w_pose", type=float, default=1.0)
  p.add_argument("--n_neg", type=int, default=0, help="random-rotation negatives per sample (0 = off)")
  p.add_argument("--predictor", default="additive", choices=["additive", "adaln"])
  p.add_argument("--ema_tau", type=float, default=0.996)
  p.add_argument("--dim", type=int, default=384)
  p.add_argument("--depth", type=int, default=6)
  p.add_argument("--pred_depth", type=int, default=4)
  p.add_argument("--eval_every", type=int, default=5000)
  p.add_argument("--eval_per_shape", type=int, default=50)
  p.add_argument("--eval_rand", type=int, default=2048)
  p.add_argument("--eval_eq", type=int, default=64)
  p.add_argument("--max_train_per_shape", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  with open(os.path.join(a.out, "args.json"), "w") as f:
    json.dump(vars(a), f, indent=2)
  print(jax.devices())

  train = SymsoltSplit(os.path.join(a.data, "train-45k"), a.color_mode, a.max_train_per_shape)
  test = SymsoltSplit(os.path.join(a.data, "test-5k"), a.color_mode, a.eval_per_shape, with_equivalents=True)
  refs = load_refs(os.path.join(a.data, "train-45k"), a.color_mode)

  model = PoseWorldModel(dim=a.dim, depth=a.depth, pred_depth=a.pred_depth, heads=a.dim // 64,
                         predictor_type=a.predictor)
  rng = np.random.default_rng(a.seed)
  b0 = sample_batch(rng, train, refs, 2, a.p_ref)
  params = model.init(jax.random.PRNGKey(a.seed), b0["xa"], b0["g_cand"][:, 0], b0["xb"])
  ema_params = params
  n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
  print(f"params: {n_params / 1e6:.1f}M")

  warmup = min(1000, a.steps // 10)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, warmup, a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.05))
  opt_state = tx.init(params)
  step_fn = make_train_step(model, tx, a.w_nce, a.w_pose, a.ema_tau)
  energy_fn = make_energy_fn(model, chunk=256)

  log_path = os.path.join(a.out, "log.jsonl")
  logf = open(log_path, "a")

  def run_eval(step):
    t0 = time.time()
    res = evaluate(energy_fn, params, ema_params, test, refs, a.eval_per_shape, a.eval_rand, a.eval_eq)
    rec = {"step": step, "eval": res, "eval_sec": time.time() - t0}
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    print(f"[eval {step}] " + json.dumps(res["mean"]), flush=True)
    return res

  run_eval(0)
  pool = ThreadPoolExecutor(2)
  fut = pool.submit(sample_batch, rng, train, refs, a.batch_size, a.p_ref, a.n_neg)
  t0 = time.time()
  for step in range(1, a.steps + 1):
    batch = fut.result()
    fut = pool.submit(sample_batch, rng, train, refs, a.batch_size, a.p_ref, a.n_neg)
    params, ema_params, opt_state, logs = step_fn(params, ema_params, opt_state, batch)
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
    f.write(flax.serialization.to_bytes({"params": params, "ema_params": ema_params}))
  print("done")


if __name__ == "__main__":
  main()
