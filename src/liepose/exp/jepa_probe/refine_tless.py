"""Latent refiner for T-LESS (FoundationPose-style iterative refinement, in DINOv2 token space).

Given a pose hypothesis g = (R, t) (LiePose parametrisation: allocentric R, scaled t), the frozen world
model predicts ẑ(g). The refiner reads [ẑ(g), z_obs, ẑ(g) - z_obs, mask weight] per patch plus a pose token
and predicts a correction (ω, Δt):  R <- R exp(ω),  t <- t + Δt.  Rotation and translation are predicted
separately (disentangled, as in FoundationPose).

Training: GT poses from the cached training crops, perturbed (mixture of small/large rotations and
translation noise). The rotation target points to the nearest symmetric equivalent of the GT, so symmetric
objects get a consistent target and no symmetry labels beyond models_info are needed.

  DATADIR=... PYTHONPATH=src python -m liepose.exp.jepa_probe.refine_tless --tokens $DATADIR/tless_tokens \
      --wm $LOGS/probe/tless_v1 --test_samples test_samples.npz --out $LOGS/probe/tless_refiner_v1
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

from liepose.exp.jepa_probe.probe import fourier_pose, pose_feat
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.probe_tless import TemplatePoseModel, TokenSplit, load_symmetries


class Refiner(nn.Module):
  dim: int = 256
  depth: int = 4
  heads: int = 8

  @nn.compact
  def __call__(self, z_hat, z_obs, fg, g):
    x = jnp.concatenate([z_hat, z_obs, z_hat - z_obs, fg[..., None]], axis=-1)
    x = nn.Dense(self.dim)(nn.LayerNorm()(x))
    pos = self.param("pos", nn.initializers.normal(0.02), (1, x.shape[1], self.dim))
    c = nn.Dense(self.dim)(nn.silu(nn.Dense(self.dim)(fourier_pose(g))))
    x = jnp.concatenate([c[:, None], x + pos], axis=1)
    for _ in range(self.depth):
      y = nn.LayerNorm()(x)
      x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads)(y, y)
      y = nn.LayerNorm()(x)
      x = x + nn.Dense(self.dim)(nn.gelu(nn.Dense(self.dim * 4)(y)))
    h = nn.LayerNorm()(x[:, 0])
    omega = nn.Dense(3, kernel_init=nn.initializers.zeros)(nn.gelu(nn.Dense(self.dim)(h)))
    dt = nn.Dense(3, kernel_init=nn.initializers.zeros)(nn.gelu(nn.Dense(self.dim)(h)))
    return omega, dt


def nearest_equiv(R, R_gt, S):
  """For each hypothesis R (B,3,3) and GT R_gt (B,3,3) with symmetry sets S (list of (M,3,3)), return R_gt·S*."""
  out = np.empty_like(R_gt)
  for i in range(len(R)):
    eq = R_gt[i] @ S[i]
    tr = np.einsum("ij,mij->m", R[i], eq)
    out[i] = eq[np.argmax(tr)]
  return out


def perturb_batch(rng, data, obj2idx, syms, batch, rot_small=15.0, rot_large=45.0, p_large=0.3, t_sigmas=(0.03, 0.1, 0.3)):
  b = rng.integers(0, len(data.obj), batch)
  R_gt, t_gt = data.R[b], data.t[b]
  ang = np.where(rng.random(batch) < p_large, rng.uniform(rot_small, rot_large, batch), rng.uniform(0, rot_small, batch))
  ax = rng.normal(size=(batch, 3))
  ax /= np.linalg.norm(ax, axis=1, keepdims=True)
  R = R_gt @ Rotation.from_rotvec(ax * np.radians(ang)[:, None]).as_matrix()
  t = t_gt + rng.normal(size=(batch, 3)) * rng.choice(t_sigmas, size=(batch, 1))
  R_tgt = nearest_equiv(R, R_gt, [syms[int(o)] for o in data.obj[b]])
  omega_tgt = Rotation.from_matrix(np.swapaxes(R, 1, 2) @ R_tgt).as_rotvec()  # R exp(ω) = R_tgt
  return {"obj": obj2idx[data.obj[b]], "tok": data.tokens[b], "fg": data.fg[b], "g": pose_feat(R, t),
          "omega": omega_tgt.astype(np.float32), "dt": (t_gt - t).astype(np.float32)}


def load_wm(wm_dir, tok_shape):
  wa = json.load(open(os.path.join(wm_dir, "args.json")))
  objs = np.load(os.path.join(wm_dir, "objects.npy"))
  wm = TemplatePoseModel(n_obj=len(objs), n_tok=tok_shape[0], tok_dim=tok_shape[1], dim=wa["dim"], depth=wa["depth"])
  init = wm.init(jax.random.PRNGKey(0), jnp.zeros((1,), jnp.int32), jnp.zeros((1, 9)))
  params = flax.serialization.from_bytes(init, open(os.path.join(wm_dir, "params.msgpack"), "rb").read())
  return wm, jax.tree_util.tree_map(jnp.asarray, params), objs


def make_refine_fn(model, wm, wm_params, use_wm=True):
  @jax.jit
  def refine(params, obj, tok, fg, R, t):
    z_hat = wm.apply(wm_params, obj, jnp.concatenate([R[:, :, 0], R[:, :, 1], t], -1)) * (1.0 if use_wm else 0.0)
    g = jnp.concatenate([R[:, :, 0], R[:, :, 1], t], -1)
    omega, dt = model.apply(params, z_hat, tok.astype(jnp.float32), fg, g)
    theta = jnp.linalg.norm(omega, axis=-1)[:, None, None] + 1e-9
    k = omega / theta[:, :, 0]
    z = jnp.zeros_like(k[:, 0])
    K = jnp.stack([jnp.stack([z, -k[:, 2], k[:, 1]], -1),
                   jnp.stack([k[:, 2], z, -k[:, 0]], -1),
                   jnp.stack([-k[:, 1], k[:, 0], z], -1)], 1)  # skew(k)
    dR = jnp.eye(3)[None] + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)  # Rodrigues
    return R @ dR, t + dt

  return refine


def evaluate_samples(refine, params, S, tok, fg, row_of, obj2idx, syms, iters=(0, 1, 2, 3), chunk=512):
  """Refine every test hypothesis; report errors per iteration for the random sample and the oracle sample."""
  N, K = S["R"].shape[:2]
  R = S["R"].reshape(-1, 3, 3).astype(np.float32)
  t = S["t"].reshape(-1, 3).astype(np.float32)
  obj = np.repeat([obj2idx[int(o)] for o in S["obj"]], K)
  rows = np.repeat([row_of[int(i)] for i in S["index"]], K)
  Rg = np.repeat(S["R_gt"], K, 0)
  tg = np.repeat(S["t_gt"], K, 0)
  Sy = [syms[int(o)] for o in np.repeat(S["obj"], K)]
  res = {}
  for it in range(max(iters) + 1):
    if it in iters:
      tr = np.stack([np.einsum("ij,mij->m", R[i], Rg[i] @ Sy[i]).max() for i in range(len(R))])
      re = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).reshape(N, K)
      te = np.linalg.norm(t - tg, axis=-1).reshape(N, K)
      res[it] = (re, te)
    if it == max(iters):
      break
    for i0 in range(0, len(R), chunk):
      sl = slice(i0, i0 + chunk)
      Rn, tn = refine(params, jnp.asarray(obj[sl]), jnp.asarray(tok[rows[sl]]), jnp.asarray(fg[rows[sl]], jnp.float32),
                      jnp.asarray(R[sl]), jnp.asarray(t[sl]))
      R[sl], t[sl] = np.asarray(Rn), np.asarray(tn)
  return res, R.reshape(N, K, 3, 3), t.reshape(N, K, 3)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--tokens", required=True)
  p.add_argument("--wm", required=True)
  p.add_argument("--test_samples", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--steps", type=int, default=40000)
  p.add_argument("--batch", type=int, default=128)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--w_t", type=float, default=1.0)
  p.add_argument("--eval_every", type=int, default=10000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--no_wm", action="store_true", help="ablation: refiner sees only z_obs + pose (world-model prediction zeroed)")
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  syms = load_symmetries(os.path.join(os.environ["DATADIR"], "bop_datasets/tless/models_eval/models_info.json"))

  train = TokenSplit(a.tokens, "train")
  te_tok = np.load(os.path.join(a.tokens, "test_tokens.npy"), mmap_mode="r")
  te_tok = np.ascontiguousarray(te_tok)
  te_fg = np.load(os.path.join(a.tokens, "test_fg.npy")).astype(np.float32)
  row_of = {int(ix): r for r, ix in enumerate(np.load(os.path.join(a.tokens, "test_meta.npz"))["index"])}
  S = dict(np.load(a.test_samples))
  wm, wm_params, objs = load_wm(a.wm, train.tokens.shape[1:])
  obj2idx = np.full(objs.max() + 1, -1, np.int64)
  obj2idx[objs] = np.arange(len(objs))

  model = Refiner()
  params = model.init(jax.random.PRNGKey(a.seed), jnp.zeros((1, 64, 384)), jnp.zeros((1, 64, 384)), jnp.zeros((1, 64)), jnp.zeros((1, 9)))
  tx = optax.chain(optax.clip_by_global_norm(1.0),
                   optax.adamw(optax.warmup_cosine_decay_schedule(0.0, a.lr, min(1000, a.steps // 10), a.steps, a.lr * 0.05), weight_decay=0.05))
  opt_state = tx.init(params)

  @jax.jit
  def step(params, opt_state, batch):
    def loss_fn(p):
      z_hat = wm.apply(wm_params, batch["obj"], batch["g"]) * (0.0 if a.no_wm else 1.0)
      om, dt = model.apply(p, z_hat, batch["tok"].astype(jnp.float32), batch["fg"], batch["g"])
      lr_ = jnp.abs(om - batch["omega"]).sum(-1).mean()
      lt = jnp.abs(dt - batch["dt"]).sum(-1).mean()
      return lr_ + a.w_t * lt, (lr_, lt)
    (loss, (lr_, lt)), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss, lr_, lt

  refine = make_refine_fn(model, wm, wm_params, use_wm=not a.no_wm)
  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  rng = np.random.default_rng(a.seed)
  pool = ThreadPoolExecutor(2)
  fut = pool.submit(perturb_batch, rng, train, obj2idx, syms, a.batch)
  t0 = time.time()
  for it in range(1, a.steps + 1):
    batch = fut.result()
    fut = pool.submit(perturb_batch, rng, train, obj2idx, syms, a.batch)
    params, opt_state, loss, lr_, lt = step(params, opt_state, {k: jnp.asarray(v) for k, v in batch.items()})
    if it % 1000 == 0:
      rec = {"it": it, "loss": float(loss), "rot_l1": float(lr_), "tran_l1": float(lt), "it_per_s": 1000 / (time.time() - t0)}
      t0 = time.time()
      logf.write(json.dumps(rec) + "\n"); logf.flush()
      print(json.dumps(rec), flush=True)
    if it % a.eval_every == 0 or it == a.steps:
      res, _, _ = evaluate_samples(refine, params, S, te_tok, te_fg, row_of, obj2idx, syms)
      for k_, (re, te) in res.items():
        msg = (f"[test it {it}] refine x{k_}: sample0 R mean {re[:, 0].mean():.2f} med {np.median(re[:, 0]):.2f} "
               f">30 {100 * (re[:, 0] > 30).mean():.2f}% | t mean {te[:, 0].mean():.4f} med {np.median(te[:, 0]):.4f} "
               f"| all-sample R med {np.median(re):.2f}")
        print(msg, flush=True)
        logf.write(json.dumps({"it": it, "iters": k_, "R_mean": float(re[:, 0].mean()), "R_med": float(np.median(re[:, 0])),
                               "t_mean": float(te[:, 0].mean()), "t_med": float(np.median(te[:, 0]))}) + "\n"); logf.flush()
      with open(os.path.join(a.out, "params.msgpack"), "wb") as f:
        f.write(flax.serialization.to_bytes(params))
  print("done", flush=True)


if __name__ == "__main__":
  main()
