"""Probe v7: retrieval-referenced, geometry-conditioned world model on frozen DINOv2 tokens.

v3-v6 predict every pose from one fixed canonical render, so the predictor must imagine the object from scratch
and only learns at training GT poses ~4 deg apart. v7 changes what it predicts from:

1. Reference = the nearest training image. Each observation gets one reference image, retrieved by allocentric
   rotation (+ lam * translation) among all training images and their symmetric equivalents, queried at a noisy
   copy of the GT pose (~ a diffusion sample) and never the observation itself. The predictor gets the reference
   tokens and pose and warps them by the relative pose: a ~1-2 deg interpolation instead of an extrapolation.
2. Geometry (--geo): every reference token gets the image displacement its patch centre undergoes from the
   reference pose to the candidate pose (plane at the object's depth, exact intrinsics, high-frequency Fourier
   features in token units), and 7 virtual object-frame points (centre + axis tips at a nominal radius; no CAD
   model) are projected at both poses and added as tokens. A 0.5 deg rotation becomes a known ~0.3 px input shift.
3. In-plane rolls (--rolls): image rotations are exact poses, extra regression targets at p + delta
   (scripts/oracle/roll_encode.py), alongside the rendered pairs (--pairs).

During training, the in-loop eval is the refinement-ceiling pick error on ceiling.npz (as ceiling_wm.py).

  python -m liepose.exp.jepa_probe.probe_ret --cache $DATADIR/dino_cache --out $LOGS/probe/v7 \
      --pairs $DATADIR/pairs --rolls $DATADIR/rolls --geo --init $LOGS/probe/v6_pair \
      --ceiling $DATADIR/oracle/ceiling.npz
"""

import argparse
from typing import Any
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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from liepose.data.symsolt.dataset import SHAPE_NAMES
from liepose.data.symsolt.presets import preset_dict
from liepose.exp.jepa_probe.probe import AdaLNBlock, fourier_pose, wxyz_to_mat
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, CachedSplit, PairSet, pool_tokens, sample_batch, weighted_energy

# SYMSOL-T camera (synth.Synthesizer with width 224, hfov 60, x 5.25): object at (R, t) sits at t + (0, 0, CAM_Z) in
# the camera frame; u = CAM_F X / Z + CAM_C. Checked against renders.
CAM_Z = 5.25 / np.tan(np.radians(30.0))
CAM_F = 112.0 * CAM_Z / 5.25
CAM_C = 112.0
OBJ_R = 1.75  # nominal object radius (shape_size / 2) for the virtual points


# ------------------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------------------

def allo_rot(R, t):
  """Allocentric rotation: undo the rotation that takes the optical axis to the ray through the object centre."""
  d = t + np.array([0.0, 0.0, CAM_Z])
  d = d / np.linalg.norm(d, axis=-1, keepdims=True)
  ax = np.cross([0.0, 0.0, 1.0], d)
  s = np.linalg.norm(ax, axis=-1, keepdims=True)
  Q = Rotation.from_rotvec((ax / np.maximum(s, 1e-12) * np.arcsin(np.clip(s, 0, 1))).reshape(-1, 3)).as_matrix()
  return np.swapaxes(Q.reshape(R.shape), -1, -2) @ R


def noisy_pose(rng, R, t, rot_deg, tran_std):
  n = len(R)
  ax = rng.normal(size=(n, 3))
  ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
  ang = np.radians(np.abs(rng.normal(0, rot_deg, size=(n, 1))))
  return Rotation.from_rotvec(ax * ang).as_matrix() @ R, t + rng.normal(0, tran_std, size=t.shape)


class RefBank:
  """Training images (and, with sym, their symmetric equivalents R S) indexed by allocentric rotation + lam t."""

  def __init__(self, data, lam=0.05, sym=True, max_sym=60):
    t0 = time.time()
    self.lam, self.tree, self.img, self.R, self.t = lam, {}, {}, {}, {}
    for s, idx in data.by_shape.items():
      S = preset_dict[SHAPE_NAMES[s]].get_symmetric_poses(np.eye(4))[:, :3, :3] if sym else np.eye(3)[None]
      S = S[:: max(1, len(S) // max_sym)]
      R = (data.R[idx][:, None] @ S[None]).reshape(-1, 3, 3)
      t = np.repeat(data.t[idx], len(S), axis=0)
      self.img[s], self.R[s], self.t[s] = np.repeat(idx, len(S)), R, t
      self.tree[s] = cKDTree(self.key(R, t))
    print(f"ref bank: {sum(len(v) for v in self.img.values())} entries in {time.time() - t0:.0f}s", flush=True)

  def key(self, R, t):
    return np.concatenate([allo_rot(R, t).reshape(len(R), 9), self.lam * t], axis=1)

  def query(self, shape, R, t, exclude):
    """Nearest entry per query that is not image `exclude` (-1: none). Returns image index, its pose (as the
    symmetric equivalent closest to the query)."""
    img, Rr, tr = np.zeros(len(R), int), np.zeros_like(R), np.zeros_like(t)
    for s in np.unique(shape):
      m = np.nonzero(shape == s)[0]
      _, j = self.tree[s].query(self.key(R[m], t[m]), k=8)
      ok = self.img[s][j] != exclude[m, None]
      j = j[np.arange(len(m)), ok.argmax(1)]
      img[m], Rr[m], tr[m] = self.img[s][j], self.R[s][j], self.t[s][j]
    return img, Rr, tr

  def query_k(self, shape, R, t, k):
    """k nearest entries per query with distinct images (an image's symmetric copies count once): (n,k) image
    indices, (n,k,3,3), (n,k,3)."""
    img, Rr, tr = np.zeros((len(R), k), int), np.zeros((len(R), k, 3, 3)), np.zeros((len(R), k, 3))
    for s in np.unique(shape):
      m = np.nonzero(shape == s)[0]
      _, j = self.tree[s].query(self.key(R[m], t[m]), k=16 * k)
      for r, jj in zip(m, j):
        _, first = np.unique(self.img[s][jj], return_index=True)
        sel = jj[np.sort(first)[:k]]
        img[r], Rr[r], tr[r] = self.img[s][sel], self.R[s][sel], self.t[s][sel]
    return img, Rr, tr

  def find(self, shape, R, t):
    """Index of the training image at exactly this pose (-1 if none): what to exclude for pair / roll anchors."""
    out = np.full(len(R), -1)
    for s in np.unique(shape):
      m = np.nonzero(shape == s)[0]
      d, j = self.tree[s].query(self.key(R[m], t[m]))
      out[m] = np.where(d < 1e-4, self.img[s][j], -1)
    return out


def rot_from_feat(g):
  """pose_feat's first two rotation columns (+ t) back to (R, t)."""
  c0, c1 = g[..., 0:3], g[..., 3:6]
  return np.stack([c0, c1, np.cross(c0, c1)], axis=-1), g[..., 6:9]


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

def fourier(x, lo, hi):
  f = 2.0 ** jnp.arange(lo, hi + 1) * jnp.pi
  y = x[..., None] * f
  return jnp.concatenate([x, jnp.sin(y).reshape(*x.shape[:-1], -1), jnp.cos(y).reshape(*x.shape[:-1], -1)], -1)


def to_cam(R, t, X):
  """object-frame points X (K,3) or (...,K,3) at pose (R (...,3,3), t (...,3)) -> camera frame (...,K,3)"""
  return jnp.einsum("...ij,...kj->...ki", R, jnp.broadcast_to(X, R.shape[:-2] + X.shape[-2:])) + (
    t + jnp.array([0.0, 0.0, CAM_Z]))[..., None, :]


def project(P):
  return CAM_F * P[..., :2] / P[..., 2:3] + CAM_C


def to_crop(u, k):
  """image pixels (...,K,2) -> crop pixels with crop k = (s, x0, y0) (...,3); k None = no crop"""
  return u if k is None else k[..., None, :1] * (u - k[..., None, 1:])


def from_crop(u, k):
  return u if k is None else u / k[..., None, :1] + k[..., None, 1:]


def geometry(Rr, tr, Rg, tg, grid, kr=None, kg=None):
  """Per-token plane-induced displacement and per-virtual-point projections, reference pose -> candidate pose.
  Reference tokens live in crop kr, the predicted (target) tokens in crop kg (crop_encode.py); None = full image."""
  patch = 224.0 / grid
  c = (jnp.arange(grid) + 0.5) * patch
  pc = jnp.stack(jnp.meshgrid(c, c, indexing="xy"), -1).reshape(-1, 2)  # token k = row * grid + col -> (u, v)
  p = from_crop(jnp.broadcast_to(pc, tr.shape[:-1] + pc.shape), kr)  # reference token centres, image pixels
  Zr = (tr[..., 2] + CAM_Z)[..., None, None]
  Pc = jnp.concatenate([(p - CAM_C) / CAM_F * Zr, jnp.broadcast_to(Zr, p.shape[:-1] + (1,))], -1)
  o = jnp.einsum("...ji,...kj->...ki", Rr, Pc - (tr + jnp.array([0.0, 0.0, CAM_Z]))[..., None, :])  # Rr^T (P - tc)
  Pg = to_cam(Rg, tg, o)
  tok = jnp.concatenate([jnp.broadcast_to(pc / 112.0 - 1.0, Pg.shape[:-1] + (2,)), (to_crop(project(Pg), kg) - pc) / patch,
                         jnp.log(Pg[..., 2:3] / Zr)], -1)
  V = OBJ_R * jnp.concatenate([jnp.zeros((1, 3)), jnp.eye(3), -jnp.eye(3)])
  Vr, Vg = to_cam(Rr, tr, V), to_cam(Rg, tg, V)
  ur, ug = to_crop(project(Vr), kr), to_crop(project(Vg), kg)
  vpt = jnp.concatenate([ur / 112.0 - 1.0, ug / 112.0 - 1.0, (ug - ur) / patch, Vr[..., 2:] / CAM_Z - 1.0,
                         Vg[..., 2:] / CAM_Z - 1.0], -1)
  return tok, vpt


def mat_feat(R, t):
  return jnp.concatenate([jnp.swapaxes(R[..., :, :2], -1, -2).reshape(*R.shape[:-2], 6), t], -1)


class GeoPredictor(nn.Module):
  """probe.PosePredictor (same layer names, so v4-v6 weights load) with the reference pose in the conditioning
  and, with geo, displacement features on every token plus virtual-point tokens."""
  dim: int
  depth: int
  heads: int
  geo: bool
  geo_fn: Any = None  # geometry(...) replacement for other cameras (sp_geometry.SpeedPlusGeometry)

  @nn.compact
  def __call__(self, z, Rr, tr, Rg, tg, kr=None, kg=None):
    g = jnp.concatenate([mat_feat(Rg, tg), mat_feat(Rg @ jnp.swapaxes(Rr, -1, -2), tg - tr)], -1)
    # auto names in PosePredictor: the outer Dense is constructed first (Dense_0), the inner one is Dense_1
    c = nn.Dense(self.dim, name="Dense_0")(nn.silu(nn.Dense(self.dim, name="Dense_1")(fourier_pose(g))))
    x = nn.Dense(self.dim, name="proj_in")(z)
    pre = [c[:, None]]
    if self.geo:
      grid = int(round(np.sqrt(z.shape[1])))
      tok, vpt = (self.geo_fn or geometry)(Rr, tr, Rg, tg, grid, kr, kg)
      x = x + nn.Dense(self.dim, name="geo_tok", kernel_init=nn.initializers.zeros)(
        nn.gelu(nn.Dense(self.dim, name="geo_tok_h")(fourier(tok, -2, 6))))
      v = nn.Dense(self.dim, name="vpt_tok")(nn.gelu(nn.Dense(self.dim, name="vpt_tok_h")(fourier(vpt, -2, 6))))
      pre.append(v + self.param("vpt_id", nn.initializers.normal(0.02), (vpt.shape[-2], self.dim)))
    n_pre = sum(t.shape[1] for t in pre)
    x = jnp.concatenate(pre + [x], axis=1)
    for i in range(self.depth):
      x = AdaLNBlock(self.dim, self.heads, name=f"block_{i}")(x, c)
    x = nn.LayerNorm(name="LayerNorm_0")(x[:, n_pre:])
    return nn.Dense(self.dim, name="proj_out")(x)


class RetPoseModel(nn.Module):
  dim: int = 384
  depth: int = 6
  heads: int = 6
  out_dim: int = 384
  geo: bool = True
  resid: bool = False  # output = learned change + a * reference tokens (a starts at 0: same function as without)
  geo_fn: Any = None

  def setup(self):
    self.predictor = GeoPredictor(dim=self.dim, depth=self.depth, heads=self.heads, geo=self.geo, geo_fn=self.geo_fn)
    self.proj = nn.Dense(self.out_dim)
    self.log_tau = self.param("log_tau", nn.initializers.constant(-2.0), ())
    if self.resid:
      self.ref_gain = self.param("ref_gain", nn.initializers.zeros, ())

  def __call__(self, z_ref, Rr, tr, Rg, tg, kr=None, kg=None):
    out = self.proj(self.predictor(z_ref, Rr, tr, Rg, tg, kr, kg))
    return out + self.ref_gain * z_ref if self.resid else out


def init_model(model, n_tok, out_dim, seed=0, init=None, k=None):
  """k: an example crop meta (1, m) for models whose geometry needs one (both reference and target)"""
  z = jnp.zeros((1, n_tok, out_dim))
  e = jnp.eye(3)[None]
  params = model.init(jax.random.PRNGKey(seed), z, e, jnp.zeros((1, 3)), e, jnp.zeros((1, 3)),
                      *(() if k is None else (jnp.asarray(k), jnp.asarray(k))))
  if init:
    src = flax.serialization.msgpack_restore(open(os.path.join(init, "params.msgpack"), "rb").read())
    flat, sflat = flax.traverse_util.flatten_dict(flax.core.unfreeze(params)), flax.traverse_util.flatten_dict(src)
    hit = [k for k in flat if k in sflat and sflat[k].shape == flat[k].shape]
    for k in hit:
      flat[k] = jnp.asarray(sflat[k])
    print(f"init from {init}: {len(hit)}/{len(flat)} tensors; new/reshaped: "
          f"{sorted({'/'.join(k[:-1]) for k in flat if k not in hit})}", flush=True)
    params = flax.traverse_util.unflatten_dict(flat)
  return params


# ------------------------------------------------------------------------------
# Batches
# ------------------------------------------------------------------------------

class Sampler:
  def __init__(self, train, bank, pairsets, a, crop=None):
    self.train, self.bank, self.a, self.crop = train, bank, a, crop  # crop: (n_train, 3) or None
    self.pairsets = pairsets
    self.excl = [bank.find(ps.label, ps.R[:, 0], ps.t[:, 0]) for ps in pairsets]
    for ps, e in zip(pairsets, self.excl):
      print(f"pair anchors found in the bank (excluded as reference): {(e >= 0).mean():.3f} of {len(e)}", flush=True)

  def refs(self, rng, shape, R, t, exclude):
    Rq, tq = noisy_pose(rng, R, t, self.a.ref_noise_deg, self.a.ref_noise_t)
    img, Rr, tr = self.bank.query(shape, Rq, tq, exclude)
    return self.train.tokens[img], Rr.astype(np.float32), tr.astype(np.float32), img

  def __call__(self, rng):
    a = self.a
    b = sample_batch(rng, self.train, a.batch_size, a.n_uni, a.n_near, a.near_min_deg, a.near_max_deg,
                     a.n_tuni, a.n_tnear, a.tnear_std)
    R, t = rot_from_feat(b.pop("g_cand"))
    b["Rc"], b["tc"] = R.astype(np.float32), t.astype(np.float32)
    i = b.pop("idx")
    b["zr"], b["Rr"], b["tr"], img = self.refs(rng, b["shape"], self.train.R[i], self.train.t[i], i)
    if self.crop is not None:
      b["kr"], b["kg"] = self.crop[img], self.crop[i]
    for k, (ps, excl) in enumerate(zip(self.pairsets, self.excl)):
      j = rng.integers(0, len(ps.label), a.pair_batch)
      zr, Rr, tr, img = self.refs(rng, ps.label[j], ps.R[j, 0], ps.t[j, 0], excl[j])
      if self.crop is not None:
        b.update({f"p{k}_kr": self.crop[img], f"p{k}_kg": ps.crop[j]})
      b.update({f"p{k}_zr": zr, f"p{k}_Rr": Rr, f"p{k}_tr": tr, f"p{k}_Rc": ps.R[j].astype(np.float32),
                f"p{k}_tc": ps.t[j].astype(np.float32), f"p{k}_tgt": ps.tokens[j], f"p{k}_fg": ps.fg[j],
                f"p{k}_small": (ps.rot_off[j, 1:] <= 8.0).astype(np.float32)})
    return b


def make_train_step(model, tx, n_pairsets, w_rec, soft_sigma, sigma_t, w_pair, w_diff=0.0, w_rank=0.0):

  def predict(params, zr, Rr, tr, Rc, tc, kr=None, kg=None):
    B, C = Rc.shape[:2]
    rep = lambda x: jnp.repeat(x, C, axis=0)
    if kr is not None:
      kr, kg = rep(kr), (kg.reshape(B * C, 3) if kg.ndim == 3 else rep(kg))
    pred = model.apply(params, rep(zr.astype(jnp.float32)), rep(Rr), rep(tr), Rc.reshape(B * C, 3, 3), tc.reshape(B * C, 3),
                       kr, kg)
    return pred.reshape(B, C, *pred.shape[1:])

  def loss_fn(params, b):
    pred = predict(params, b["zr"], b["Rr"], b["tr"], b["Rc"], b["tc"], b.get("kr"), b.get("kg"))
    e = weighted_energy(pred, b["tgt"].astype(jnp.float32)[:, None], (FG_FLOOR + b["fg"])[:, None])
    tau = jnp.exp(params["params"]["log_tau"])
    logit_t = -0.5 * (b["d_gt"] / soft_sigma) ** 2 - 0.5 * (b["dt_gt"] / sigma_t) ** 2
    pose_nce = optax.softmax_cross_entropy(-e / tau, jax.nn.softmax(logit_t, axis=-1)).mean()
    rec = e[:, 0].mean()
    loss = pose_nce + w_rec * rec
    logs = {"pose_nce": pose_nce, "rec": rec, "tau": tau, "pose_acc": (jnp.argmin(e, axis=1) == 0).mean()}
    for k in range(n_pairsets):
      pp = predict(params, b[f"p{k}_zr"], b[f"p{k}_Rr"], b[f"p{k}_tr"], b[f"p{k}_Rc"], b[f"p{k}_tc"], b.get(f"p{k}_kr"),
                   b.get(f"p{k}_kg"))
      tgt, fg = b[f"p{k}_tgt"].astype(jnp.float32), b[f"p{k}_fg"]
      pair = weighted_energy(pp, tgt, FG_FLOOR + fg).mean()
      loss = loss + w_pair * pair
      logs[f"pair{k}"] = pair
      # candidate 0 is the anchor (GT render / unrolled image); the others are small offsets of it
      w = FG_FLOOR + jnp.maximum(fg[:, :1], fg[:, 1:])
      small = b[f"p{k}_small"]  # diff / rank only on offsets <= 8 deg (the wide rolls would dominate the sums)
      gap = small * jax.lax.stop_gradient(weighted_energy(tgt[:, 1:], tgt[:, :1], w))  # exact-render energy gap
      if w_diff > 0:  # the change between candidates, relative to the true change
        d_err = weighted_energy(pp[:, 1:] - pp[:, :1], tgt[:, 1:] - tgt[:, :1], w)
        diff = (small * d_err).sum() / (gap.sum() + 1e-8)
        loss = loss + w_diff * diff
        logs[f"diff{k}"] = diff
      if w_rank > 0:  # anchor as the observation: E(offset) - E(anchor) should reach the exact-render gap
        e = weighted_energy(pp, tgt[:, :1], FG_FLOOR + fg[:, :1])
        rank = (small * jax.nn.relu(gap - (e[:, 1:] - e[:, :1]))).sum() / (gap.sum() + 1e-8)
        loss = loss + w_rank * rank
        logs[f"rank{k}"] = rank
    logs["loss"] = loss
    return loss, logs

  @jax.jit
  def step(params, opt_state, b):
    grads, logs = jax.grad(loss_fn, has_aux=True)(params, b)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, logs

  return step


# ------------------------------------------------------------------------------
# Refinement-ceiling evaluation (same candidates and metric as scripts/oracle/ceiling_wm.py)
# ------------------------------------------------------------------------------

def make_energy_fn(model):
  @jax.jit
  def energy(params, zr, Rr, tr, tgt, fg, Rc, tc, kr=None, kg=None):
    """one observation: reference (P,D) + pose, target (P,D), fg (P,), candidates (N,3,3), (N,3) -> (N,)"""
    n = Rc.shape[0]
    if kr is not None:
      kr, kg = jnp.broadcast_to(kr, (n, 3)), jnp.broadcast_to(kg, (n, 3))
    pred = model.apply(params, jnp.broadcast_to(zr.astype(jnp.float32), (n,) + zr.shape), jnp.broadcast_to(Rr, (n, 3, 3)),
                       jnp.broadcast_to(tr, (n, 3)), Rc, tc, kr, kg)
    return weighted_energy(pred, tgt.astype(jnp.float32)[None], (FG_FLOOR + fg)[None])

  return energy


class Ceiling:
  """ceiling.npz candidates around 500 test-5k observations; one reference per observation, retrieved at a noisy
  GT (fixed seed) and shared by all its candidates."""

  def __init__(self, path, cache, pool, bank, train, a, crop=None, obs=None):
    C = np.load(path)
    self.rot_err, self.tran_err, self.kind, self.label = C["rot_err"], C["tran_err"], C["kind"], C["label"]
    self.Rc, self.tc = C["R"].astype(np.float32), C["t"].astype(np.float32)
    idx = C["index"]
    if crop is not None:  # observations cropped around their own foreground (crop_encode.py ceiling)
      o = np.load(obs or os.path.join(cache, "ceiling_obs.npz"))  # make_holdout.py: held-out training images
      assert (o["index"] == idx).all()
      self.z, self.w, self.kg = o["tokens"], o["fg"].astype(np.float32), o["crop"]
    else:
      tok = np.load(os.path.join(cache, "test-5k_tokens.npy"), mmap_mode="r")
      fg = np.load(os.path.join(cache, "test-5k_fg.npy"), mmap_mode="r")
      self.z, self.w = pool_tokens(np.asarray(tok[idx]), np.asarray(fg[idx]), pool)
    Rq, tq = noisy_pose(np.random.default_rng(123), C["R"][:, 0], C["t"][:, 0], a.ref_noise_deg, a.ref_noise_t)
    img, Rr, tr = bank.query(self.label, Rq, tq, np.full(len(idx), -1))
    self.zr, self.Rr, self.tr = train.tokens[img], Rr.astype(np.float32), tr.astype(np.float32)
    self.kr = None if crop is None else crop[img]
    ang = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", allo_rot(Rr, tr), allo_rot(C["R"][:, 0], C["t"][:, 0])) - 1) / 2, -1, 1)))
    self.ref_dist_deg = ang
    print(f"ceiling: {len(idx)} obs, reference-to-GT allocentric angle median {np.median(ang):.2f} deg", flush=True)

  def energies(self, energy_fn, params):
    return np.stack([np.asarray(energy_fn(params, self.zr[i], self.Rr[i], self.tr[i], self.z[i], self.w[i].astype(np.float32),
                                          self.Rc[i], self.tc[i], *(() if self.kr is None else (self.kr[i], self.kg[i]))))
                     for i in range(len(self.z))])

  def summary(self, E):
    rot = self.kind == "rot"
    sel = np.array([self.rot_err[i, np.nonzero(rot[i])[0][np.argmin(E[i, rot[i]])]] for i in range(len(E))])
    disc = {}
    for lo, hi in [(0.2, 0.75), (0.75, 1.5), (1.5, 3.0), (3.0, 9.0)]:
      m = rot & (self.rot_err >= lo) & (self.rot_err < hi)
      disc[f"{lo}-{hi}"] = float((E[m] > np.broadcast_to(E[:, :1], E.shape)[m]).mean())
    tr = self.kind == "tran"
    return {"pick_rot_err": float(sel.mean()),
            "per_shape": {SHAPE_NAMES[s]: float(sel[self.label == s].mean()) for s in np.unique(self.label)},
            "P(E_pert>E_gt) rot": disc,
            "P(E_pert>E_gt) tran": float((E[tr] > np.broadcast_to(E[:, :1], E.shape)[tr]).mean())}


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--cache", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--ceiling", required=True)
  p.add_argument("--ceiling_obs", default=None, help="observations of --ceiling (default: <cache>/ceiling_obs.npz)")
  p.add_argument("--pool", type=int, default=2)
  p.add_argument("--train_split", default="train-225k")
  p.add_argument("--geo", action="store_true")
  p.add_argument("--crop", action="store_true", help="--cache/--pairs/--rolls are zoomed crops (crop_encode.py)")
  p.add_argument("--no_sym", action="store_true", help="retrieve among the images only, not their symmetric equivalents")
  p.add_argument("--lam", type=float, default=0.05, help="translation weight in the retrieval key")
  p.add_argument("--ref_noise_deg", type=float, default=3.0, help="retrieval query = GT rotated by |N(0, this)| deg")
  p.add_argument("--ref_noise_t", type=float, default=0.03)
  p.add_argument("--pairs", default=None)
  p.add_argument("--rolls", default=None)
  p.add_argument("--w_pair", type=float, default=1.0)
  p.add_argument("--w_diff", type=float, default=0.0, help="match predicted to true token change between pair candidates")
  p.add_argument("--w_rank", type=float, default=0.0, help="energy gap anchor->offset must reach the exact-render gap")
  p.add_argument("--resid", action="store_true", help="output = change + learned gain * reference tokens")
  p.add_argument("--pair_batch", type=int, default=8)
  p.add_argument("--steps", type=int, default=20000)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--n_uni", type=int, default=11)
  p.add_argument("--n_near", type=int, default=12)
  p.add_argument("--near_min_deg", type=float, default=0.0)
  p.add_argument("--near_max_deg", type=float, default=30.0)
  p.add_argument("--n_tuni", type=int, default=4)
  p.add_argument("--n_tnear", type=int, default=4)
  p.add_argument("--tnear_std", type=float, default=0.1)
  p.add_argument("--w_rec", type=float, default=1.0)
  p.add_argument("--soft_sigma", type=float, default=15.0)
  p.add_argument("--sigma_t", type=float, default=0.1)
  p.add_argument("--init", default=None)
  p.add_argument("--lr", type=float, default=2e-4)
  p.add_argument("--dim", type=int, default=384)
  p.add_argument("--depth", type=int, default=6)
  p.add_argument("--eval_every", type=int, default=5000)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  print(jax.devices(), flush=True)

  train = CachedSplit(a.cache, a.train_split, a.pool)
  bank = RefBank(train, a.lam, not a.no_sym)
  pairsets = [PairSet(d, a.pool) for d in (a.pairs, a.rolls) if d]
  crop = None
  if a.crop:
    crop = np.load(os.path.join(a.cache, f"{a.train_split}_p{a.pool}_meta.npz"))["crop"]
    for ps, d in zip(pairsets, [d for d in (a.pairs, a.rolls) if d]):
      ps.crop = np.load(os.path.join(d, "pairs_meta.npz"))["crop"]
  ceil = Ceiling(a.ceiling, a.cache, a.pool, bank, train, a, crop, a.ceiling_obs)
  model = RetPoseModel(dim=a.dim, depth=a.depth, heads=a.dim // 64, out_dim=train.tokens.shape[-1], geo=a.geo,
                       resid=a.resid)
  params = init_model(model, train.tokens.shape[1], train.tokens.shape[-1], a.seed, a.init)
  print(f"params: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.1f}M", flush=True)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(1000, a.steps // 10), a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.05))
  opt_state = tx.init(params)
  step_fn = make_train_step(model, tx, len(pairsets), a.w_rec, a.soft_sigma, a.sigma_t, a.w_pair, a.w_diff, a.w_rank)
  energy_fn = make_energy_fn(model)
  sampler = Sampler(train, bank, pairsets, a, crop)
  logf = open(os.path.join(a.out, "log.jsonl"), "a")

  def run_eval(step):
    t0 = time.time()
    E = ceil.energies(energy_fn, params)
    rec = {"step": step, "ceiling": ceil.summary(E), "eval_sec": time.time() - t0}
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    print(f"[eval {step}] " + json.dumps(rec["ceiling"]), flush=True)
    return E

  rng = np.random.default_rng(a.seed)
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
      if step % 500 == 0:
        print(json.dumps(logs), flush=True)
    if step % a.eval_every == 0 or step == a.steps:
      E = run_eval(step)
  np.save(os.path.join(a.out, "ceiling_energy.npy"), E)
  with open(os.path.join(a.out, "params.msgpack"), "wb") as f:
    f.write(flax.serialization.to_bytes(params))
  print("done", flush=True)


if __name__ == "__main__":
  main()
