"""World model as learned 3D feature fields of the 30 T-LESS objects (field.py's SPEED+ field, ported; no CAD model).

  field        per object o: tri-planes (3 x --plane_res^2 x --plane_dim) in the object frame over a box of half-size
               r_o = --box x (models_info diameter / 2); MLP, density, latent, view head and background shared by all
               objects (object identity: the BOP task's obj_id, as the retrieval world model's bank)
  render(g)    the observation's crop camera (box meta + the image's intrinsics, TlessGeometry) and the candidate pose
               g = (allocentric R, translation_scaled, LiePose's T-LESS parametrisation) -> rays through 16 x 16
               sub-patch centres, --n_samp depths in [z_c - r_o, z_c + r_o], volume rendering, 2 x 2 pooling ->
               8 x 8 tokens in the tless_tokens space (frozen DINOv2 of LiePose's crop x visible mask)
  appearance   wm_eval.app_stats of the observation's own tokens (the same for every candidate pose)
  loss         token reconstruction at the GT pose, weighted per token by the visible mask (FG_FLOOR + fg, as the
               energy), so occluded parts of the object are not learned as background

Crop meta (N, 10) per row of the token cache: s = 224 / dzi_size, x0, y0 (crop origin, image px), bbox w, h,
focal_ratio, fx, fy, cx, cy, and the translation_scaled of the true 3D centre (LiePose's encodes the annotated 2D
centroid, off by ~2 cm in x / y on ~1/4 of the training crops): scripts/tless/crop_meta_cache.py.
Units: metres (the dataset's translations), so r_o = --box x diameter / 2000.

Eval: probe_ret_tless.Ceiling on the same held-out source images and observations as tless_ret_v1 (val_frac 0.03,
rng 1234 / 99, candidate seed 7), so the field and the retrieval world model are scored on identical candidates.
Gate, fixed before training (val, vs tless_ret_v1's 0.527 / 0.636 / 7.19 deg): P(E>E_gt) 1-3 >= 0.6,
P(E>E_gt) 3-10 >= 0.75 and pick_rot_err_mean below the retrieval world model's; checkpoint choice on val pick error.

  DATADIR=... PYTHONPATH=src python -m liepose.exp.jepa_probe.tless_field --out $LOGS/probe/tl_field_v1
"""

import argparse
import glob
import json
import os
import time
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from liepose.exp.bop.ds_config import DS
from liepose.exp.jepa_probe import probe_ret_tless as prt
from liepose.exp.jepa_probe.field import fourier
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.probe_tless import TokenSplit, load_symmetries
from liepose.exp.jepa_probe.sp_geometry import quat_to_mat
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.utils.jnpfn import get_allo_rotation

N_OBJ = DS["n_obj"]
# liepose.exp.bop.run's T-LESS translation parametrisation
TRAN_SCALE = tuple(DS["tran_scale"] or (1.0, 1.0, 1.0))  # bop.ds_config (BOP_DS); bop.run asserts they are set
TRAN_OFFSET = tuple(DS["tran_offset"] or (0.0, 0.0, 0.0))


def tless_crop_meta(crops_dir):
  """(N, 10) crop meta from export_crops.py --meta shards + index / translation_scaled for alignment checks"""
  sh = [np.load(f) for f in sorted(glob.glob(os.path.join(crops_dir, "shard_*.npz")))]
  d = {k: np.concatenate([s[k] for s in sh]) for k in ["index", "translation_scaled"] + META_KEYS}
  return crop_meta10(d), d["index"], d["translation_scaled"]


META_KEYS = ["bbox", "bbox_center", "dzi_size", "focal_ratio", "cam"]


def crop_meta10(d):
  """(N, 10) crop meta from LiePose's per-crop bbox, bbox_center, dzi_size, focal_ratio, cam (a batch or the shards)"""
  dzi, bb, K = np.asarray(d["dzi_size"], np.float64).reshape(-1), np.asarray(d["bbox"], np.float64), np.asarray(d["cam"], np.float64)
  c = np.asarray(d["bbox_center"], np.float64)
  k = np.stack([224.0 / dzi, c[:, 0] - dzi / 2, c[:, 1] - dzi / 2, np.maximum(bb[:, 2] - bb[:, 0], 1), np.maximum(bb[:, 3] - bb[:, 1], 1),
                np.asarray(d["focal_ratio"], np.float64).reshape(-1), K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], -1)
  return k.astype(np.float32)


def bilinear_obj(planes, o, uv):
  """planes (N_OBJ, R, R, C), o (N,) object index, uv (N, ..., 2) in [-1, 1] -> (N, ..., C); zero outside.
  field.bilinear per sample without gathering a whole plane per sample (points index the object's plane directly)"""
  R = planes.shape[1]
  p = (uv + 1.0) * 0.5 * (R - 1)
  p0 = jnp.floor(p)
  w = p - p0
  i0 = jnp.clip(p0.astype(jnp.int32), 0, R - 1)
  i1 = jnp.clip(i0 + 1, 0, R - 1)
  ob = o.reshape(o.shape + (1,) * (uv.ndim - 2))
  g = lambda i, j: planes[ob, i[..., 1], j[..., 0]]
  return (g(i0, i0) * ((1 - w[..., 0]) * (1 - w[..., 1]))[..., None] + g(i0, i1) * (w[..., 0] * (1 - w[..., 1]))[..., None]
          + g(i1, i0) * ((1 - w[..., 0]) * w[..., 1])[..., None] + g(i1, i1) * (w[..., 0] * w[..., 1])[..., None])


class TlessGeometry:
  """sp_geometry.SpeedPlusGeometry with per-crop intrinsics (k[..., 6:10]) and T-LESS's translation parametrisation"""

  def __init__(self):
    self.scale = jnp.array(TRAN_SCALE, jnp.float32)
    self.offset = jnp.array(TRAN_OFFSET, jnp.float32)

  def __hash__(self):  # flax module attribute
    return id(self)

  def cam_pose(self, R, ts, k):
    """(allocentric R, translation_scaled) in crop k -> (egocentric R, camera-frame t, dataset units)"""
    d = (ts + self.offset) * jnp.concatenate([k[..., 3:5], k[..., 5:6]], -1) * self.scale
    c = k[..., 1:3] + 112.0 / k[..., :1] + d[..., :2]  # bbox_center + centroid offset, image px
    z = jnp.maximum(d[..., 2:3], 1e-2)
    t = jnp.concatenate([(c - k[..., 8:10]) / k[..., 6:8] * z, z], -1)
    A = quat_to_mat(get_allo_rotation(t))  # ego -> allo
    return jnp.swapaxes(A, -1, -2) @ R, t

  @staticmethod
  def from_crop(u, k):
    return u / k[..., None, :1] + k[..., None, 1:3]

  @staticmethod
  def ray(u, k):
    """image px (N, P, 2) -> rays at camera z = 1 (N, P, 3)"""
    return jnp.concatenate([(u - k[:, None, 8:10]) / k[:, None, 6:8], jnp.ones(u.shape[:-1] + (1,))], -1)


class TlessField(nn.Module):
  radii: Any = None  # (N_OBJ,) box half-size per object, dataset units
  plane_res: int = 128
  plane_dim: int = 32
  hid: int = 128
  out_dim: int = 384
  n_samp: int = 32
  sub: int = 16

  def setup(self):
    init = nn.initializers.normal(0.1)
    self.planes = [self.param(f"plane_{n}", init, (N_OBJ, self.plane_res, self.plane_res, self.plane_dim)) for n in ("xy", "xz", "yz")]
    self.mlp = [nn.Dense(self.hid), nn.Dense(self.hid)]
    self.sigma = nn.Dense(1)
    self.app = nn.Dense(64)
    self.film = nn.Dense(2 * self.hid, kernel_init=nn.initializers.zeros)
    self.head = [nn.Dense(256), nn.Dense(256)]
    self.out = nn.Dense(self.out_dim)
    self.bg = nn.Dense(self.out_dim)

  def point(self, x, o):
    """x (N, ..., 3) object frame / r_o, o (N,) object index -> sigma (N, ...), h (N, ..., hid)"""
    f = sum(bilinear_obj(p, o, x[..., ij]) for p, ij in zip(self.planes, ([0, 1], [0, 2], [1, 2])))
    h = jnp.concatenate([f, fourier(x, 4)], -1)
    for d in self.mlp:
      h = nn.gelu(d(h))
    inside = (jnp.abs(x) <= 1.0).all(-1)
    return jnp.where(inside, nn.softplus(self.sigma(h)[..., 0] - 1.0), 0.0), h

  def __call__(self, obj, Rg, tg, kg, app, jitter=None, with_opacity=False, with_latent=False):
    """obj (N,) BOP obj_id, candidate poses (N, 3, 3) allocentric / (N, 3) translation_scaled, crop meta (N, 10),
    appearance statistics (N, 2D) -> tokens (N, 64, out_dim) [, opacity (N, 64)] [, latent (N, 64, hid)]"""
    geo, S, M = TlessGeometry(), self.sub, self.n_samp
    o = obj.astype(jnp.int32) - 1
    r = jnp.asarray(self.radii, jnp.float32)[o]  # (N,)
    R, t = geo.cam_pose(Rg, tg, kg)
    cc = (jnp.arange(S) + 0.5) * (224.0 / S)
    pc = jnp.stack(jnp.meshgrid(cc, cc, indexing="xy"), -1).reshape(-1, 2)
    d = geo.ray(geo.from_crop(jnp.broadcast_to(pc, t.shape[:-1] + pc.shape), kg), kg)  # (N, S^2, 3)
    off = (jnp.arange(M) + (0.5 if jitter is None else jitter)) / M
    z = t[:, None, None, 2] + r[:, None, None] * (2.0 * off - 1.0)
    z = jnp.broadcast_to(z, d.shape[:2] + (M,))
    P = d[:, :, None] * z[..., None]
    X = jnp.einsum("nji,nrmj->nrmi", R, P - t[:, None, None]) / r[:, None, None, None]
    sigma, h = self.point(X, o)
    dz = 2.0 * r[:, None, None] / M * jnp.linalg.norm(d, axis=-1)[..., None]
    alpha = 1.0 - jnp.exp(-sigma * dz)
    T = jnp.cumprod(jnp.concatenate([jnp.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1] + 1e-10], -1), -1)
    w = T * alpha
    opa = w.sum(-1, keepdims=True)
    h_ray = (w[..., None] * h).sum(-2) / (opa + 1e-6)
    bgw = 1.0 - opa
    a = nn.gelu(self.app(app))[:, None]
    sc, sh = jnp.split(self.film(a), 2, -1)
    vd = jnp.einsum("nji,nrj->nri", R, d / jnp.linalg.norm(d, axis=-1, keepdims=True))
    y = jnp.concatenate([h_ray * (1.0 + sc) + sh, fourier(vd, 2), jnp.broadcast_to(a, h_ray.shape[:2] + (64,))], -1)
    for l in self.head:
      y = nn.gelu(l(y))
    f = (1.0 - bgw) * self.out(y) + bgw * self.bg(a)
    N = f.shape[0]
    pool = lambda x: x.reshape(N, S // 2, 2, S // 2, 2, x.shape[-1]).mean((2, 4)).reshape(N, (S // 2) ** 2, x.shape[-1])
    if with_latent:
      return pool(f), pool(opa)[..., 0], pool(h_ray)
    if with_opacity:
      return pool(f), pool(opa)[..., 0]
    return pool(f)


def object_radii(models_info, box):
  info = json.load(open(models_info))
  return np.array([box * info[str(o)]["diameter"] / 2000.0 for o in range(1, N_OBJ + 1)], np.float32)  # mm -> m


def load_tless_field(path):
  """(model, params) of a trained field (args.json + params.msgpack in path)"""
  a = json.load(open(os.path.join(path, "args.json")))
  model = TlessField(radii=tuple(a["radii"]), plane_res=a["plane_res"], plane_dim=a["plane_dim"], hid=a["hid"], n_samp=a["n_samp"])
  params = model.init(jax.random.PRNGKey(0), jnp.ones((1,)), jnp.eye(3)[None], jnp.zeros((1, 3)), jnp.ones((1, 10)), jnp.zeros((1, 768)))
  params = flax.serialization.from_bytes(params, open(os.path.join(path, "params.msgpack"), "rb").read())
  return model, jax.tree_util.tree_map(jnp.asarray, params)


def field_energies(model, params, ceil, obs, meta):
  @jax.jit
  def energy(obj, Rc, tc, k, tgt, fg):
    n = Rc.shape[0]
    tgt = tgt.astype(jnp.float32)
    pred = model.apply(params, jnp.full((n,), obj), Rc, tc, jnp.broadcast_to(k, (n,) + k.shape), jnp.broadcast_to(app_stats(tgt), (n, 768)))
    return weighted_energy(pred, tgt[None], (FG_FLOOR + fg)[None])

  return np.stack([np.asarray(energy(int(obs.obj[i]), ceil.R[j], ceil.t[j], meta[i], obs.tokens[i], obs.fg[i])) for j, i in enumerate(ceil.rows)])


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--cache", default=None, help="token cache (default $DATADIR/tless_tokens)")
  p.add_argument("--meta", default=None, help="crop meta dir with {split}_crop_meta.npy (default: the cache)")
  p.add_argument("--steps", type=int, default=100000)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--lr", type=float, default=2e-3)
  p.add_argument("--box", type=float, default=1.2, help="box half-size r_o in units of the object radius (diameter / 2)")
  p.add_argument("--plane_res", type=int, default=128)
  p.add_argument("--plane_dim", type=int, default=32)
  p.add_argument("--n_samp", type=int, default=32)
  p.add_argument("--hid", type=int, default=128)
  p.add_argument("--eval_every", type=int, default=5000)
  p.add_argument("--liepose_t", action="store_true", help="fit at LiePose's translation_scaled (2D-centroid offset) instead")
  p.add_argument("--seed", type=int, default=0)
  # Ceiling (= tless_ret_v1's)
  p.add_argument("--val_frac", type=float, default=0.03)
  p.add_argument("--ceil_n", type=int, default=500)
  p.add_argument("--ceil_per_bin", type=int, default=16)
  p.add_argument("--ceil_uni", type=int, default=64)
  p.add_argument("--ceil_tran", type=int, default=16)
  p.add_argument("--ceil_tstd", type=float, default=0.1)
  p.add_argument("--ref_noise_deg", type=float, default=5.0)
  p.add_argument("--ref_noise_t", type=float, default=0.1)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  cache = a.cache or os.path.join(D, DS["tokens"])
  os.makedirs(a.out, exist_ok=True)
  info = os.path.join(D, DS["models_info"])
  a.radii = object_radii(info, a.box).tolist()
  json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
  train = TokenSplit(cache, "train")
  meta = np.load(os.path.join(a.meta or cache, "train_crop_meta.npy")).astype(np.float32)
  assert len(meta) == len(train.obj)
  if not a.liepose_t:  # fit (and evaluate) at the true 3D centre (scripts/tless/crop_meta_cache.py)
    train.t = np.load(os.path.join(a.meta or cache, "train_ts_true.npy")).astype(np.float64)
    assert train.t.shape == (len(train.obj), 3)
  syms = load_symmetries(info)
  prt.T_SCALE = np.array([2.0, 2.0, 1.0])  # tless_ret_v1's (Ceiling's translation candidates)
  grp = np.unique(train.index)  # held-out source images, as probe_ret_tless
  held = np.random.default_rng(1234).permutation(grp)[: int(a.val_frac * len(grp))]
  is_val = np.isin(train.index, held)
  val_rows, train_rows = np.nonzero(is_val)[0], np.nonzero(~is_val)[0]
  val_obs = np.sort(np.random.default_rng(99).choice(val_rows, min(a.ceil_n, len(val_rows)), replace=False))
  ceil = prt.Ceiling(train, val_obs, syms, None, a, 7)

  model = TlessField(radii=tuple(a.radii), plane_res=a.plane_res, plane_dim=a.plane_dim, hid=a.hid, n_samp=a.n_samp)
  params = model.init(jax.random.PRNGKey(a.seed), jnp.ones((1,)), jnp.eye(3)[None], jnp.zeros((1, 3)), jnp.asarray(meta[:1]), jnp.zeros((1, 768)))
  print(f"params: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6:.2f}M", flush=True)
  sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(1000, a.steps // 10), a.steps, a.lr * 0.05)
  tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
  opt_state = tx.init(params)
  rays = (model.sub ** 2, a.n_samp)

  @jax.jit
  def step(params, opt_state, key, obj, R, t, k, tgt, fg):
    def loss_fn(params):
      pred = model.apply(params, obj, R, t, k, app_stats(tgt), jax.random.uniform(key, (R.shape[0],) + rays))
      return weighted_energy(pred, tgt, FG_FLOOR + fg).mean()

    loss, g = jax.value_and_grad(loss_fn)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss

  logf = open(os.path.join(a.out, "log.jsonl"), "a")
  best = np.inf

  def run_eval(i):
    nonlocal best
    t0 = time.time()
    s = ceil.summary(field_energies(model, params, ceil, train, meta))
    rec = {"step": i, "val": s, "eval_sec": time.time() - t0}
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    print(f"[eval {i}] val {json.dumps(s)}", flush=True)
    if s["pick_rot_err_mean"] < best:
      best = s["pick_rot_err_mean"]
      open(os.path.join(a.out, "params.msgpack"), "wb").write(flax.serialization.to_bytes(params))

  rng = np.random.default_rng(a.seed)
  key = jax.random.PRNGKey(a.seed + 1)
  run_eval(0)
  t0 = time.time()
  for i in range(1, a.steps + 1):
    b = np.sort(rng.choice(train_rows, a.batch_size))
    key, k = jax.random.split(key)
    params, opt_state, loss = step(params, opt_state, k, jnp.asarray(train.obj[b]), jnp.asarray(train.R[b], jnp.float32),
                                   jnp.asarray(train.t[b], jnp.float32), jnp.asarray(meta[b]), jnp.asarray(train.tokens[b], jnp.float32),
                                   jnp.asarray(train.fg[b]))
    if i % 100 == 0:
      rec = {"step": i, "loss": float(loss), "it_per_s": 100 / (time.time() - t0)}
      t0 = time.time()
      logf.write(json.dumps(rec) + "\n")
      logf.flush()
      if i % 1000 == 0:
        print(json.dumps(rec), flush=True)
    if i % a.eval_every == 0:
      run_eval(i)
  print("TLESS_FIELD_DONE", flush=True)


if __name__ == "__main__":
  main()
