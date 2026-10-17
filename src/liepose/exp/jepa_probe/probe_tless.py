"""T-LESS world model: pose-conditioned predictor on frozen DINOv2 tokens of LiePose's masked crops.

Instance-level: each object has a learned 8x8 token template (initialised from the mean tokens of its
training crops) in place of SYMSOL's reference render, so ẑ(g) = P(template[obj], g). The pose g uses
LiePose's own parametrisation (allocentric rotation + scaled translation), so the energy can score
denoiser samples directly. Training is the same soft-target pose-contrastive loss as probe v3.

  PYTHONPATH=src python -m liepose.exp.jepa_probe.probe_tless --cache $DATADIR/tless_tokens --out $LOGS/probe/tless_v1
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

from liepose.exp.jepa_probe.probe import PosePredictor, geodesic_deg, pose_feat, wxyz_to_mat
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, perturb, weighted_energy


class TokenSplit:
  def __init__(self, cache, split, max_n=None):
    d = np.load(os.path.join(cache, f"{split}_meta.npz"))
    n = len(d["object_id"]) if max_n is None else min(max_n, len(d["object_id"]))
    self.tokens = np.load(os.path.join(cache, f"{split}_tokens.npy"), mmap_mode="r")[:n]
    self.tokens = np.ascontiguousarray(self.tokens)
    self.fg = np.load(os.path.join(cache, f"{split}_fg.npy"))[:n].astype(np.float32)
    self.obj = d["object_id"][:n].astype(np.int64)
    self.R = wxyz_to_mat(d["rotation_allo"][:n])
    self.t = d["translation_scaled"][:n].astype(np.float64)
    self.index = d["index"][:n] if "index" in d else np.arange(n)
    print(f"{split}: {n} crops, {len(np.unique(self.obj))} objects", flush=True)


def load_symmetries(models_info_path, max_cont=72):
  """object_id -> (M,3,3) rotations S such that R and R@S are equivalent (identity included)."""
  info = json.load(open(models_info_path))
  out = {}
  for k, v in info.items():
    syms = [np.eye(3)]
    for s in v.get("symmetries_discrete", []):
      syms.append(np.asarray(s, np.float64).reshape(4, 4)[:3, :3])
    for s in v.get("symmetries_continuous", []):
      ax = np.asarray(s["axis"], np.float64)
      syms += [Rotation.from_rotvec(ax * a).as_matrix() for a in np.linspace(0, 2 * np.pi, max_cont, endpoint=False)[1:]]
    out[int(k)] = np.stack(syms)
  return out


class TemplatePoseModel(nn.Module):
  n_obj: int
  n_tok: int
  tok_dim: int
  dim: int = 384
  depth: int = 6
  template_init: np.ndarray = None

  def setup(self):
    init = (lambda *_: jnp.asarray(self.template_init)) if self.template_init is not None else nn.initializers.normal(0.02)
    self.templates = self.param("templates", init, (self.n_obj, self.n_tok, self.tok_dim))
    self.predictor = PosePredictor(dim=self.dim, depth=self.depth, heads=self.dim // 64)
    self.proj = nn.Dense(self.tok_dim)
    self.log_tau = self.param("log_tau", nn.initializers.constant(-2.0), ())

  def __call__(self, obj_idx, g):
    return self.proj(self.predictor(self.templates[obj_idx], g))


def sample_batch(rng, data, obj2idx, batch_size, n_uni, n_near, near_max_deg):
  b = rng.integers(0, len(data.obj), batch_size)
  C = 1 + n_uni + n_near
  R = np.repeat(data.R[b][:, None], C, axis=1)
  if n_uni:
    R[:, 1 : 1 + n_uni] = Rotation.random(batch_size * n_uni, random_state=rng.integers(2**31)).as_matrix().reshape(batch_size, n_uni, 3, 3)
  if n_near:
    R[:, 1 + n_uni :] = perturb(R[:, 1 + n_uni :].reshape(-1, 3, 3), near_max_deg, 0.0, rng).reshape(batch_size, n_near, 3, 3)
  t = np.repeat(data.t[b][:, None], C, axis=1)
  tr = np.einsum("bcij,bij->bc", R, data.R[b])
  d_gt = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).astype(np.float32)
  return {"obj": obj2idx[data.obj[b]], "tgt": data.tokens[b], "fg": data.fg[b], "g_cand": pose_feat(R, t), "d_gt": d_gt}


def make_train_step(model, tx, soft_sigma, w_rec):
  def loss_fn(params, batch):
    g = batch["g_cand"]
    B, C = g.shape[:2]
    pred = model.apply(params, jnp.repeat(batch["obj"], C), g.reshape(B * C, -1))
    pred = pred.reshape(B, C, *pred.shape[1:])
    e = weighted_energy(pred, batch["tgt"].astype(jnp.float32)[:, None], (FG_FLOOR + batch["fg"])[:, None])
    tau = jnp.exp(params["params"]["log_tau"])
    target = jax.nn.softmax(-0.5 * (batch["d_gt"] / soft_sigma) ** 2, axis=-1)
    nce = optax.softmax_cross_entropy(-e / tau, target).mean()
    rec = e[:, 0].mean()
    loss = nce + w_rec * rec
    return loss, {"loss": loss, "nce": nce, "rec": rec, "tau": tau, "pose_acc": (jnp.argmin(e, 1) == 0).mean()}

  @jax.jit
  def step(params, opt_state, batch):
    grads, logs = jax.grad(loss_fn, has_aux=True)(params, batch)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, logs

  return step


def evaluate(model, params, data, obj2idx, syms, n_eval, n_rand, chunk=256, seed=0):
  """Rank GT vs n_rand random rotations (GT translation); report top-1 / top-1% distance to the symmetry set."""
  rng = np.random.default_rng(seed)
  R_rand = Rotation.random(n_rand, random_state=seed).as_matrix()

  @jax.jit
  def energies(obj, g, tgt, fg):
    pred = model.apply(params, jnp.full((g.shape[0],), obj), g)
    return weighted_energy(pred, tgt[None].astype(jnp.float32), (FG_FLOOR + fg)[None])

  idx = rng.choice(len(data.obj), min(n_eval, len(data.obj)), replace=False)
  top1, prec, chance, pct = [], [], [], []
  for i in idx:
    R_all = np.concatenate([data.R[i][None], R_rand])
    pad = (-len(R_all)) % chunk
    R_all = np.concatenate([R_all, R_rand[:pad]])
    g = pose_feat(R_all, np.broadcast_to(data.t[i], (len(R_all), 3)))
    e = np.concatenate([np.asarray(energies(int(obj2idx[data.obj[i]]), g[j : j + chunk], data.tokens[i], data.fg[i]))
                        for j in range(0, len(g), chunk)])
    e_gt, e_r = e[0], e[1 : 1 + n_rand]
    eq = data.R[i][None] @ syms[int(data.obj[i])]
    d_eq = geodesic_deg(R_rand, eq)
    order = np.argsort(e_r)
    k = max(1, n_rand // 100)
    pct.append((e_r < e_gt).mean() + 0.5 * (e_r == e_gt).mean())  # ties count half
    top1.append(d_eq[order[0]])
    prec.append((d_eq[order[:k]] < 30).mean())
    chance.append((d_eq < 30).mean())
  return {"gt_percentile": float(np.mean(pct)), "gt_in_top5pct": float(np.mean(np.array(pct) < 0.05)),
          "top1_err_deg_median": float(np.median(top1)), "top1pct_within30": float(np.mean(prec)),
          "chance_within30": float(np.mean(chance)), "lift": float(np.mean(prec) / max(np.mean(chance), 1e-9))}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--cache", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--models_info", default=None, help="BOP models_info.json for symmetries; 'none' = no symmetries (SPEED+)")
  p.add_argument("--test_split", default="test")
  p.add_argument("--steps", type=int, default=30000)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--n_uni", type=int, default=15)
  p.add_argument("--n_near", type=int, default=16)
  p.add_argument("--near_max_deg", type=float, default=30.0)
  p.add_argument("--soft_sigma", type=float, default=15.0)
  p.add_argument("--w_rec", type=float, default=1.0)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--dim", type=int, default=384)
  p.add_argument("--depth", type=int, default=6)
  p.add_argument("--eval_every", type=int, default=5000)
  p.add_argument("--eval_n", type=int, default=300)
  p.add_argument("--eval_rand", type=int, default=2048)
  p.add_argument("--max_train", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  models_info = a.models_info or os.path.join(os.environ["DATADIR"], "bop_datasets/tless/models_eval/models_info.json")
  syms = None if a.models_info == "none" else load_symmetries(models_info)

  train = TokenSplit(a.cache, "train", a.max_train)
  test = TokenSplit(a.cache, a.test_split)
  objs = np.unique(train.obj)
  obj2idx = np.full(objs.max() + 1, -1, np.int64)
  obj2idx[objs] = np.arange(len(objs))
  if syms is None:
    syms = {int(o): np.eye(3)[None] for o in objs}
  np.save(os.path.join(a.out, "objects.npy"), objs)
  tmpl = np.stack([train.tokens[train.obj == o][:2000].astype(np.float32).mean(0) for o in objs])

  model = TemplatePoseModel(n_obj=len(objs), n_tok=tmpl.shape[1], tok_dim=tmpl.shape[2], dim=a.dim, depth=a.depth,
                            template_init=tmpl)
  params = model.init(jax.random.PRNGKey(a.seed), jnp.zeros((1,), jnp.int32), jnp.zeros((1, 9)))
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(1000, a.steps // 10), a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.05))
  opt_state = tx.init(params)
  step_fn = make_train_step(model, tx, a.soft_sigma, a.w_rec)
  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  best = -1.0

  def run_eval(step):
    nonlocal best
    res = {"step": step, "test": evaluate(model, params, test, obj2idx, syms, a.eval_n, a.eval_rand),
           "train": evaluate(model, params, train, obj2idx, syms, a.eval_n // 3, a.eval_rand)}
    logf.write(json.dumps(res) + "\n")
    logf.flush()
    print(f"[eval {step}] test {json.dumps(res['test'])}", flush=True)
    if res["test"]["top1pct_within30"] > best:
      best = res["test"]["top1pct_within30"]
      with open(os.path.join(a.out, "params.msgpack"), "wb") as f:
        f.write(flax.serialization.to_bytes(params))

  rng = np.random.default_rng(a.seed)
  run_eval(0)
  pool = ThreadPoolExecutor(1)
  fut = pool.submit(sample_batch, rng, train, obj2idx, a.batch_size, a.n_uni, a.n_near, a.near_max_deg)
  t0 = time.time()
  for step in range(1, a.steps + 1):
    batch = fut.result()
    fut = pool.submit(sample_batch, rng, train, obj2idx, a.batch_size, a.n_uni, a.n_near, a.near_max_deg)
    params, opt_state, logs = step_fn(params, opt_state, batch)
    if step % 100 == 0:
      logs = {k: float(v) for k, v in logs.items()}
      logs.update(step=step, it_per_s=100 / (time.time() - t0))
      t0 = time.time()
      logf.write(json.dumps(logs) + "\n")
      logf.flush()
      if step % 1000 == 0:
        print(json.dumps(logs), flush=True)
    if step % a.eval_every == 0 or step == a.steps:
      run_eval(step)
  print("done", flush=True)


if __name__ == "__main__":
  main()
