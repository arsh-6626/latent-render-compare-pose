"""T-LESS port of the v7 world model (probe_ret), stage 1: retrieved references, no geometry tokens.

The T-LESS crops (export_crops.py / encode_crops.py, LiePose's own masking, crop and pose parametrisation) are
already object-centred, which was the main fix on SYMSOL-T. The geometry tokens need per-crop intrinsics and
metric translation, which the token cache does not store, so this stage keeps geo off:

  ẑ(g) = P(z_ref, g_ref -> g)  with  z_ref = tokens of the nearest training crop of the same object to g
                                       (allocentric rotation, symmetric equivalents from models_info)

Poses are LiePose's: allocentric rotation (rotation_allo) and translation_scaled, divided by T_SCALE inside the
model so the Fourier pose features see O(1) values. Retrieval uses the rotation (lam * normalised translation).

Loss: soft-target pose contrast as probe_ret, but with symmetry-aware candidate distances (the GT's whole
symmetry set) and translation candidates. Checkpoint chosen on held-out training crops; the test split is
evaluated once with the chosen weights.

Eval (one reference per observation, retrieved at a noisy GT): candidates = GT + rotation perturbations in
1-3 / 3-10 / 10-30 deg + uniform rotations + translation perturbations; pick error = symmetric rotation error of
the min-energy rotation candidate; P(E_pert > E_gt) per bin. --template evaluates a probe_tless (learned per-
object template) model on the same candidates instead.

  PYTHONPATH=src python -m liepose.exp.jepa_probe.probe_ret_tless --cache $DATADIR/tless_tokens \
      --out $LOGS/probe/tless_ret_v1
  PYTHONPATH=src python -m liepose.exp.jepa_probe.probe_ret_tless --cache $DATADIR/tless_tokens \
      --out $LOGS/probe/tless_v1_ceil --template $LOGS/probe/tless_v1
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from liepose.exp.jepa_probe.probe import pose_feat
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, perturb, weighted_energy
from liepose.exp.jepa_probe.probe_ret import RetPoseModel, init_model
from liepose.exp.jepa_probe.probe_tless import TemplatePoseModel, TokenSplit, load_symmetries

T_SCALE = np.array([2.0, 2.0, 1.0])  # translation_scaled -> O(1) (test |t| 90th pct ~1.8, 1.7, 1.5)
BINS = [(1.0, 3.0), (3.0, 10.0), (10.0, 30.0)]


def sym_deg(R, R_gt, S):
  """R (...,3,3) vs the symmetry set R_gt S (M,3,3) -> (...) min angle, deg"""
  eq = R_gt[None] @ S
  tr = np.einsum("...ij,mij->...m", R, eq)
  return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min(-1)


def random_rot(rng, n):
  return Rotation.random(n, random_state=rng.integers(1 << 31)).as_matrix()


class TlessBank:
  """Training crops (and their symmetric equivalents R S) per object, keyed by allocentric R (+ lam t_n).
  Exclusion is by source image (data.index): augmented copies of one image (SPEED+ --copies 2) count as one."""

  def __init__(self, data, rows, syms, lam, max_sym=36):
    t0 = time.time()
    self.lam, self.tree, self.img, self.grp, self.R, self.t = lam, {}, {}, {}, {}, {}
    for o in np.unique(data.obj[rows]):
      idx = rows[data.obj[rows] == o]
      S = syms[int(o)]
      S = S[:: max(1, int(np.ceil(len(S) / max_sym)))]
      self.R[o] = (data.R[idx][:, None] @ S[None]).reshape(-1, 3, 3)
      self.t[o] = np.repeat(data.t[idx], len(S), axis=0)
      self.img[o] = np.repeat(idx, len(S))
      self.grp[o] = data.index[self.img[o]]
      self.tree[o] = cKDTree(self.key(self.R[o], self.t[o]))
    print(f"ref bank: {sum(len(v) for v in self.img.values())} entries, {len(self.tree)} objects in "
          f"{time.time() - t0:.0f}s", flush=True)

  def key(self, R, t):
    return np.concatenate([R.reshape(len(R), 9), self.lam * t / T_SCALE], axis=1)

  def query(self, obj, R, t, exclude):
    """nearest entry per query not from source image `exclude` (a data.index value; -1: none) -> crop row, its
    (symmetric) pose"""
    img, Rr, tr = np.zeros(len(R), int), np.zeros_like(R), np.zeros_like(t)
    for o in np.unique(obj):
      m = np.nonzero(obj == o)[0]
      _, j = self.tree[o].query(self.key(R[m], t[m]), k=8)
      ok = self.grp[o][j] != exclude[m, None]
      j = j[np.arange(len(m)), ok.argmax(1)]
      img[m], Rr[m], tr[m] = self.img[o][j], self.R[o][j], self.t[o][j]
    return img, Rr, tr


def noisy(rng, R, t, deg, tstd):
  ax = rng.normal(size=(len(R), 3))
  ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
  ang = np.radians(np.abs(rng.normal(0, deg, size=(len(R), 1))))
  return Rotation.from_rotvec(ax * ang).as_matrix() @ R, t + rng.normal(0, tstd, size=t.shape) * T_SCALE


class Sampler:
  """[GT | n_uni uniform R | n_near near R | n_tnear near t | n_tuni other crops' t] per training crop."""

  def __init__(self, data, rows, bank, syms, a):
    self.data, self.rows, self.bank, self.syms, self.a = data, rows, bank, syms, a

  def __call__(self, rng):
    a, d = self.a, self.data
    b = self.rows[rng.integers(0, len(self.rows), a.batch_size)]
    C = 1 + a.n_uni + a.n_near + a.n_tnear + a.n_tuni
    R = np.repeat(d.R[b][:, None], C, axis=1)
    t = np.repeat(d.t[b][:, None], C, axis=1)
    B = a.batch_size
    o = 1
    R[:, o : o + a.n_uni] = random_rot(rng, B * a.n_uni).reshape(B, a.n_uni, 3, 3)
    o += a.n_uni
    R[:, o : o + a.n_near] = perturb(R[:, o : o + a.n_near].reshape(-1, 3, 3), a.near_max_deg, 0.0, rng).reshape(B, a.n_near, 3, 3)
    o += a.n_near
    t[:, o : o + a.n_tnear] += rng.normal(0, a.tnear_std, size=(B, a.n_tnear, 3)) * T_SCALE
    o += a.n_tnear
    t[:, o:] = d.t[self.rows[rng.integers(0, len(self.rows), (B, a.n_tuni))]]
    d_gt = np.stack([sym_deg(R[i], d.R[b[i]], self.syms[int(d.obj[b[i]])]) for i in range(B)]).astype(np.float32)
    dt_gt = np.linalg.norm((t - d.t[b][:, None]) / T_SCALE, axis=-1).astype(np.float32)
    Rq, tq = noisy(rng, d.R[b], d.t[b], a.ref_noise_deg, a.ref_noise_t)
    img, Rr, tr = self.bank.query(d.obj[b], Rq, tq, d.index[b])
    return {"zr": d.tokens[img], "Rr": Rr.astype(np.float32), "tr": (tr / T_SCALE).astype(np.float32),
            "Rc": R.astype(np.float32), "tc": (t / T_SCALE).astype(np.float32), "tgt": d.tokens[b], "fg": d.fg[b],
            "d_gt": d_gt, "dt_gt": dt_gt}


def make_train_step(model, tx, soft_sigma, sigma_t, w_rec):

  def loss_fn(params, b):
    B, C = b["Rc"].shape[:2]
    rep = lambda x: jnp.repeat(x, C, axis=0)
    pred = model.apply(params, rep(b["zr"].astype(jnp.float32)), rep(b["Rr"]), rep(b["tr"]), b["Rc"].reshape(B * C, 3, 3),
                       b["tc"].reshape(B * C, 3)).reshape(B, C, -1, b["tgt"].shape[-1])
    e = weighted_energy(pred, b["tgt"].astype(jnp.float32)[:, None], (FG_FLOOR + b["fg"])[:, None])
    tau = jnp.exp(params["params"]["log_tau"])
    target = jax.nn.softmax(-0.5 * (b["d_gt"] / soft_sigma) ** 2 - 0.5 * (b["dt_gt"] / sigma_t) ** 2, axis=-1)
    nce = optax.softmax_cross_entropy(-e / tau, target).mean()
    rec = e[:, 0].mean()
    loss = nce + w_rec * rec
    return loss, {"loss": loss, "nce": nce, "rec": rec, "tau": tau, "pose_acc": (jnp.argmin(e, 1) == 0).mean()}

  @jax.jit
  def step(params, opt_state, b):
    grads, logs = jax.grad(loss_fn, has_aux=True)(params, b)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, logs

  return step


class Ceiling:
  """Fixed candidate sets around n observations (rows of `data`), identical for every model given the seed."""

  def __init__(self, data, rows, syms, bank, a, seed, exclude_self=False):
    rng = np.random.default_rng(seed)
    self.rows = rows
    n_bin, n_uni, n_t = a.ceil_per_bin, a.ceil_uni, a.ceil_tran
    R, t, kind, err, terr = [], [], [], [], []
    for i in rows:
      Rg, tg, S = data.R[i], data.t[i], syms[int(data.obj[i])]
      Rs = [Rg[None]] + [perturb(np.repeat(Rg[None], n_bin, 0), hi, lo, rng) for lo, hi in BINS] + [random_rot(rng, n_uni)]
      Rs = np.concatenate(Rs + [np.repeat(Rg[None], n_t, 0)])
      ts = np.repeat(tg[None], len(Rs), 0)
      ts[-n_t:] += rng.normal(0, a.ceil_tstd, size=(n_t, 3)) * T_SCALE
      R.append(Rs)
      t.append(ts)
      err.append(sym_deg(Rs, Rg, S))
      terr.append(np.linalg.norm((ts - tg) / T_SCALE, axis=-1))
    self.R, self.t = np.stack(R).astype(np.float32), np.stack(t).astype(np.float32)
    self.err, self.terr = np.stack(err), np.stack(terr)
    nr = 1 + len(BINS) * n_bin + n_uni
    self.kind = np.array(["gt"] + sum([[f"bin{k}"] * n_bin for k in range(len(BINS))], []) + ["uni"] * n_uni + ["tran"] * n_t)
    self.is_rot = np.arange(len(self.kind)) < nr
    Rq, tq = noisy(rng, data.R[rows], data.t[rows], a.ref_noise_deg, a.ref_noise_t)
    self.img, self.Rr, self.tr = bank.query(data.obj[rows], Rq, tq, data.index[rows] if exclude_self else np.full(len(rows), -1)) if bank is not None else (None, None, None)
    if bank is not None:
      d = np.array([sym_deg(self.Rr[k][None], data.R[i], syms[int(data.obj[i])])[0] for k, i in enumerate(rows)])
      print(f"ceiling: {len(rows)} obs, reference-to-GT angle median {np.median(d):.1f} deg", flush=True)

  def summary(self, E):
    """ties (e.g. an untrained model) count half and are broken at random, never in favour of the GT"""
    rot = self.is_rot
    jit = np.random.default_rng(0).random(E.shape)
    pick = np.array([self.err[i, rot][np.lexsort((jit[i, rot], E[i, rot]))[0]] for i in range(len(E))])
    gt_worse = lambda m: float(((E > E[:, :1]) + 0.5 * (E == E[:, :1]))[m].mean())
    res = {"pick_rot_err_mean": float(pick.mean()), "pick_rot_err_median": float(np.median(pick)),
           "pick_gt15": float((pick > 15).mean())}
    near = rot & (self.kind != "gt") & (self.kind != "uni")
    for lo, hi in BINS:  # by the candidate's symmetric error (a perturbation can land near a symmetric equivalent)
      m = near[None] & (self.err >= lo) & (self.err < hi)
      res[f"P(E>E_gt) {lo:g}-{hi:g}"] = gt_worse(m)
    m = self.kind == "uni"
    res["gt_rank_among_uniform"] = 1.0 - gt_worse(np.broadcast_to(m, E.shape))
    res["uniform_top1_err_median"] = float(np.median(self.err[np.arange(len(E)), np.where(m)[0][np.argmin(E[:, m], 1)]]))
    res["P(E>E_gt) tran"] = gt_worse(np.broadcast_to(self.kind == "tran", E.shape))
    return res


def save_energies(out, name, ceil, E):
  """per-candidate energies, for metrics computed outside (e.g. scripts/speedplus/esa_score.py)"""
  np.savez(os.path.join(out, f"energies_{name}.npz"), rows=ceil.rows, R=ceil.R, t=ceil.t, E=E, kind=ceil.kind,
           is_rot=ceil.is_rot)


def ret_energies(model, params, ceil, refs, obs):
  """refs: the split the bank indexes (train), obs: the split of the observations"""
  @jax.jit
  def energy(zr, Rr, tr, tgt, fg, Rc, tc):
    n = Rc.shape[0]
    pred = model.apply(params, jnp.broadcast_to(zr.astype(jnp.float32), (n,) + zr.shape), jnp.broadcast_to(Rr, (n, 3, 3)),
                       jnp.broadcast_to(tr, (n, 3)), Rc, tc)
    return weighted_energy(pred, tgt.astype(jnp.float32)[None], (FG_FLOOR + fg)[None])

  ts = T_SCALE.astype(np.float32)
  return np.stack([np.asarray(energy(refs.tokens[ceil.img[k]], ceil.Rr[k].astype(np.float32), (ceil.tr[k] / ts).astype(np.float32),
                                     obs.tokens[i], obs.fg[i], ceil.R[k], ceil.t[k] / ts))
                   for k, i in enumerate(ceil.rows)])


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--cache", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--models_info", default=None, help="BOP models_info.json; 'none' = no symmetries (SPEED+)")
  p.add_argument("--t_scale", type=float, nargs=3, default=[2.0, 2.0, 1.0],
                 help="translation_scaled -> O(1); T-LESS 2 2 1, SPEED+ 1 1 1")
  p.add_argument("--val_split", default=None, help="checkpoint-selection split; default: --val_frac of train held out")
  p.add_argument("--test_splits", nargs="+", default=["test"])
  p.add_argument("--template", default=None, help="evaluate this probe_tless model on the same candidates and exit")
  p.add_argument("--init", default=None, help="probe_ret checkpoint to start from (matching tensors only)")
  p.add_argument("--lam", type=float, default=0.0)
  p.add_argument("--max_sym", type=int, default=36)
  p.add_argument("--ref_noise_deg", type=float, default=5.0)
  p.add_argument("--ref_noise_t", type=float, default=0.1, help="normalised units (x T_SCALE)")
  p.add_argument("--steps", type=int, default=30000)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--n_uni", type=int, default=11)
  p.add_argument("--n_near", type=int, default=12)
  p.add_argument("--near_max_deg", type=float, default=30.0)
  p.add_argument("--n_tnear", type=int, default=4)
  p.add_argument("--n_tuni", type=int, default=4)
  p.add_argument("--tnear_std", type=float, default=0.2)
  p.add_argument("--soft_sigma", type=float, default=10.0)
  p.add_argument("--sigma_t", type=float, default=0.1)
  p.add_argument("--w_rec", type=float, default=1.0)
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--dim", type=int, default=384)
  p.add_argument("--depth", type=int, default=6)
  p.add_argument("--val_frac", type=float, default=0.03)
  p.add_argument("--ceil_n", type=int, default=500)
  p.add_argument("--ceil_per_bin", type=int, default=16)
  p.add_argument("--ceil_uni", type=int, default=64)
  p.add_argument("--ceil_tran", type=int, default=16)
  p.add_argument("--ceil_tstd", type=float, default=0.1)
  p.add_argument("--eval_every", type=int, default=5000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--eval_only", action="store_true", help="re-run the final eval of the checkpoint in --out")
  a = p.parse_args()
  global T_SCALE
  T_SCALE = np.array(a.t_scale)
  os.makedirs(a.out, exist_ok=True)
  if not a.eval_only:
    json.dump({**vars(a), "T_SCALE": T_SCALE.tolist()}, open(os.path.join(a.out, "args.json"), "w"), indent=2)
  print(jax.devices(), flush=True)
  train = TokenSplit(a.cache, "train")
  tests = {s: TokenSplit(a.cache, s) for s in a.test_splits}
  if a.models_info == "none":
    syms = {int(o): np.eye(3)[None] for o in np.unique(train.obj)}
  else:
    syms = load_symmetries(a.models_info or os.path.join(os.environ["DATADIR"], "bop_datasets/tless/models_eval/models_info.json"))
  ev = np.random.default_rng(99)
  if a.val_split:  # a separate split: the whole train split is the bank
    val, train_rows = TokenSplit(a.cache, a.val_split), np.arange(len(train.obj))
    val_obs = np.sort(ev.choice(len(val.obj), min(a.ceil_n, len(val.obj)), replace=False))
  else:  # held-out source images of the train split (fixed, independent of --seed)
    grp = np.unique(train.index)
    held = np.random.default_rng(1234).permutation(grp)[: int(a.val_frac * len(grp))]
    is_val = np.isin(train.index, held)
    val, val_rows, train_rows = train, np.nonzero(is_val)[0], np.nonzero(~is_val)[0]
    np.save(os.path.join(a.out, "val_rows.npy"), val_rows)
    val_obs = np.sort(ev.choice(val_rows, min(a.ceil_n, len(val_rows)), replace=False))
  test_obs = {s: np.sort(ev.choice(len(d.obj), min(a.ceil_n, len(d.obj)), replace=False)) for s, d in tests.items()}

  if a.template:  # learned-template baseline on identical candidates (no reference)
    ta = json.load(open(os.path.join(a.template, "args.json")))
    objs = np.load(os.path.join(a.template, "objects.npy"))
    obj2idx = np.full(objs.max() + 1, -1, np.int64)
    obj2idx[objs] = np.arange(len(objs))
    model = TemplatePoseModel(n_obj=len(objs), n_tok=train.tokens.shape[1], tok_dim=train.tokens.shape[2], dim=ta["dim"],
                              depth=ta["depth"])
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1,), jnp.int32), jnp.zeros((1, 9)))
    params = jax.tree_util.tree_map(jnp.asarray, flax.serialization.from_bytes(
      params, open(os.path.join(a.template, "params.msgpack"), "rb").read()))

    @jax.jit
    def energy(obj, g, tgt, fg):
      pred = model.apply(params, obj, g)
      return weighted_energy(pred, tgt.astype(jnp.float32)[None], (FG_FLOOR + fg)[None])

    res = {}
    for k, (name, data, rows) in enumerate([("val", val, val_obs)] + [(s, tests[s], test_obs[s]) for s in a.test_splits]):
      seed = 7 + k
      ceil = Ceiling(data, rows, syms, None, a, seed)
      E = np.stack([np.asarray(energy(jnp.full((ceil.R.shape[1],), int(obj2idx[data.obj[i]])), pose_feat(ceil.R[k], ceil.t[k]), data.tokens[i], data.fg[i]))
                    for k, i in enumerate(rows)])
      res[name] = ceil.summary(E)
      save_energies(a.out, name, ceil, E)
      print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(a.out, "ceiling.json"), "w"), indent=2)
    return

  bank = TlessBank(train, train_rows, syms, a.lam, a.max_sym)
  model = RetPoseModel(dim=a.dim, depth=a.depth, heads=a.dim // 64, out_dim=train.tokens.shape[-1], geo=False)
  params = init_model(model, train.tokens.shape[1], train.tokens.shape[-1], a.seed, a.init)
  print(f"params: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.1f}M", flush=True)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(1000, a.steps // 10), a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.05))
  opt_state = tx.init(params)
  step_fn = make_train_step(model, tx, a.soft_sigma, a.sigma_t, a.w_rec)
  sampler = Sampler(train, train_rows, bank, syms, a)
  ceil_val = Ceiling(val, val_obs, syms, bank, a, 7)  # held-out val images are not in the bank

  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  best = np.inf

  def run_eval(step):
    nonlocal best
    t0 = time.time()
    s = ceil_val.summary(ret_energies(model, params, ceil_val, train, val))
    rec = {"step": step, "val": s, "eval_sec": time.time() - t0}
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    print(f"[eval {step}] val {json.dumps(s)}", flush=True)
    if s["pick_rot_err_mean"] < best:
      best = s["pick_rot_err_mean"]
      with open(os.path.join(a.out, "params.msgpack"), "wb") as f:
        f.write(flax.serialization.to_bytes(params))

  rng = np.random.default_rng(a.seed)
  if a.eval_only:
    a.steps = 0
  else:
    run_eval(0)
  ex = ThreadPoolExecutor(1)
  fut = ex.submit(sampler, rng)
  t0 = time.time()
  for step in range(1, a.steps + 1):
    batch = fut.result()
    fut = ex.submit(sampler, rng)
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

  # the chosen checkpoint, once on test
  params = flax.serialization.from_bytes(params, open(os.path.join(a.out, "params.msgpack"), "rb").read())
  E = ret_energies(model, params, ceil_val, train, val)
  res = {"val": ceil_val.summary(E)}
  save_energies(a.out, "val", ceil_val, E)
  for k, s in enumerate(a.test_splits):
    c = Ceiling(tests[s], test_obs[s], syms, bank, a, 8 + k)
    E = ret_energies(model, params, c, train, tests[s])
    res[s] = c.summary(E)
    save_energies(a.out, s, c, E)
  print("final", json.dumps(res), flush=True)
  json.dump(res, open(os.path.join(a.out, "ceiling_eval.json" if a.eval_only else "ceiling.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
