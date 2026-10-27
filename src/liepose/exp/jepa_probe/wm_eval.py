"""Evaluation harness for SPEED+ world models (from world model v2's wm2.py; the v2-v4 models themselves are not
part of this repo: all three failed their gates, see results/speedplus_diag/wm{2,3,4}_*).

  Bank          K nearest training crops from distinct source images (a model may ignore them, e.g. field.py)
  Eval          fixed candidate sets on n observations of a split: GT, rotations at 0.25-1 / 1-3 / 3-10 / 10-30 deg,
                the three 180-degree twins, uniform rotations, 1-3 deg roll / tilt, and a refiner estimate 1-3 deg
                away; per-candidate and shared (references retrieved once at the estimate) protocols;
                summary(): P(E_pert > E_gt) per bin, GT error / appearance distance, pick error
  app_stats     per-channel mean / std of a crop's tokens (the appearance code input)
  app_distance  median clean-vs-colour-augmented token distance on val (the error yardstick)
"""

import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from liepose.exp.jepa_probe import probe_ret_tless as prt
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, perturb, weighted_energy
from liepose.exp.jepa_probe.probe_tless import TokenSplit


TWIN = np.stack([np.diag([1.0 if j == i else -1.0 for j in range(3)]) for i in range(3)])  # pi about body x, y, z


BINS = [(0.25, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 30.0)]


def app_stats(z):
  """(..., P, D) tokens -> (..., 2D) per-channel mean and std over tokens"""
  return jnp.concatenate([z.mean(-2), z.std(-2)], -1)


class Bank(prt.TlessBank):
  def query_k(self, R, t, exclude, K):
    """K nearest entries from distinct source images, none from `exclude` (data.index; -1: none) -> rows (n,K), poses"""
    o = next(iter(self.tree))
    _, j = self.tree[o].query(self.key(R, t), k=8 * K)
    img = np.zeros((len(R), K), int)
    for n in range(len(R)):
      seen, out = {int(exclude[n])}, []
      for jj in j[n]:
        g = int(self.grp[o][jj])
        if g not in seen:
          seen.add(g)
          out.append(jj)
          if len(out) == K:
            break
      img[n] = out
    return self.img[o][img], self.R[o][img], self.t[o][img]


class Eval:
  """Fixed candidate sets on n observations of a split; every candidate has its own K references (sampling)."""

  def __init__(self, obs, meta, bank, a, seed):
    rng = np.random.default_rng(seed)
    self.obs, self.meta = obs, meta
    self.rows = np.sort(rng.choice(len(obs.R), min(a.eval_n, len(obs.R)), replace=False))
    R, t = obs.R[self.rows], obs.t[self.rows]
    Rs = [R[:, None]] + [perturb(np.repeat(R, a.eval_per_bin, 0), hi, lo, rng).reshape(len(R), a.eval_per_bin, 3, 3) for lo, hi in BINS]
    Rs += [R[:, None] @ TWIN[None], Rotation.random(len(R) * 16, random_state=rng.integers(1 << 31)).as_matrix().reshape(len(R), 16, 3, 3)]
    # 1-3 deg about the allocentric viewing axis (roll; what the roll pairs supervise) and about axes orthogonal to it
    m = a.eval_per_bin
    ang = np.radians(rng.uniform(1, 3, (len(R) * m, 1))) * rng.choice([-1, 1], (len(R) * m, 1))
    Rs.append((Rotation.from_rotvec(np.array([0, 0, 1.0]) * ang).as_matrix() @ np.repeat(R, m, 0)).reshape(len(R), m, 3, 3))
    ax = np.concatenate([rng.normal(size=(len(R) * m, 2)), np.zeros((len(R) * m, 1))], 1)
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    Rs.append((Rotation.from_rotvec(ax * np.abs(ang)).as_matrix() @ np.repeat(R, m, 0)).reshape(len(R), m, 3, 3))
    # the refiner's current estimate, 1-3 deg from the true pose (last candidate)
    Re = perturb(R, 3.0, 1.0, rng)
    Rs.append(Re[:, None])
    self.Rc = np.concatenate(Rs, 1).astype(np.float32)
    self.tc = np.repeat(t[:, None], self.Rc.shape[1], 1).astype(np.float32)
    self.kind = np.array(["gt"] + sum([[f"{lo:g}-{hi:g}"] * a.eval_per_bin for lo, hi in BINS], []) + ["twin"] * 3 + ["uni"] * 16
                         + ["roll 1-3"] * m + ["tilt 1-3"] * m + ["est"])
    self.ang = np.degrees(np.arccos(np.clip((np.einsum("bij,bcij->bc", R, self.Rc) - 1) / 2, -1, 1)))
    n, C = self.Rc.shape[:2]
    img, Rr, tr = bank.query_k(self.Rc.reshape(-1, 3, 3).astype(np.float64), self.tc.reshape(-1, 3).astype(np.float64),
                               np.full(n * C, -1), a.n_ref)
    K = a.n_ref
    self.img, self.Rr, self.tr = img.reshape(n, C, K), Rr.reshape(n, C, K, 3, 3).astype(np.float32), tr.reshape(n, C, K, 3).astype(np.float32)
    # shared protocol (refiner step): K references retrieved once at the estimate and used for every candidate; only
    # "P est" = P(E(estimate) > E(true pose)) is reported for it: any pull of the energy towards the references' pose
    # favours the estimate, not the true pose (a GT-centred candidate set would favour the true pose instead)
    img, Rr, tr = bank.query_k(Re, t, np.full(n, -1), K)
    rep = lambda x: np.repeat(x[:, None], C, 1)
    self.shared = (rep(img), rep(Rr.astype(np.float32)), rep(tr.astype(np.float32)))
    # the same with the references retrieved at the true pose: a world model that learned "the references' pose is
    # wrong" (instead of comparing) prefers the estimate here
    img, Rr, tr = bank.query_k(R, t, np.full(n, -1), K)
    self.shared_gt = (rep(img), rep(Rr.astype(np.float32)), rep(tr.astype(np.float32)))

  def energies(self, model, params, train, train_meta, shared=False):
    @jax.jit
    def energy(zr, Rr, tr, kr, Rc, tc, kg, tgt, fg):
      C = Rc.shape[0]
      pred = model.apply(params, zr.astype(jnp.float32), Rr, tr, kr, Rc, tc, jnp.broadcast_to(kg, (C,) + kg.shape),
                         jnp.broadcast_to(app_stats(tgt), (C, 2 * tgt.shape[-1])))
      return weighted_energy(pred, tgt[None], (FG_FLOOR + fg)[None])

    I, RR, TR = {False: (self.img, self.Rr, self.tr), True: self.shared, "gt": self.shared_gt}[shared]
    return np.stack([np.asarray(energy(train.tokens[I[k]], RR[k], TR[k], train_meta[I[k]], self.Rc[k],
                                       self.tc[k], self.meta[i], self.obs.tokens[i].astype(np.float32), self.obs.fg[i]))
                     for k, i in enumerate(self.rows)])

  def summary(self, E, d_app, shared=False):
    if shared:
      return {"P est": float((E[:, self.kind == "est"] > E[:, :1]).mean()),
              "gt_err_over_app": float(np.median(np.sqrt(E[:, 0])) / d_app)}
    res = {}
    for lo, hi in BINS:
      m = self.kind == f"{lo:g}-{hi:g}"
      res[f"P {lo:g}-{hi:g}"] = float((E[:, m] > E[:, :1]).mean())
    for k in ["twin", "roll 1-3", "tilt 1-3", "est"]:
      res[f"P {k}"] = float((E[:, self.kind == k] > E[:, :1]).mean())
    res["gt_err_over_app"] = float(np.median(np.sqrt(E[:, 0])) / d_app)
    rot = ~np.isin(self.kind, ["uni", "roll 1-3", "tilt 1-3", "est"])
    res["pick_err_median"] = float(np.median(self.ang[np.arange(len(E)), np.where(rot)[0][np.argmin(E[:, rot], 1)]]))
    return res


def app_distance(D):
  """median clean-vs-augmented token distance on the val crops that were augmented (val_aug)"""
  v, va = TokenSplit(os.path.join(D, "speedplus_tokens"), "val"), TokenSplit(os.path.join(D, "speedplus_tokens"), "val_aug")
  shards = [np.load(f) for f in sorted(glob.glob(os.path.join(D, "speedplus_crops", "val_aug", "shard_*.npz")))]
  if "color_aug" in shards[0].files:
    on, idx = np.concatenate([x["color_aug"] for x in shards]), np.concatenate([x["index"] for x in shards])
  else:  # meta-only shards (HPC copy): the flags are uploaded separately
    f = np.load(os.path.join(D, "speedplus_crops", "val_aug_color_aug.npz"))
    on, idx = f["color_aug"], f["index"]
  assert (idx == va.index).all(), "color_aug flags misaligned with val_aug tokens"
  r = np.flatnonzero(on)[:2000]
  return float(np.median(np.sqrt(((va.tokens[r].astype(np.float32) - v.tokens[r].astype(np.float32)) ** 2).mean((1, 2)))))
