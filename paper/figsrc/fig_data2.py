"""Real inference for the second set of figures (read-only use of the trained checkpoints; no training, no tuning).

Models as fig_data.py: joint (LaRC, T-LESS GT crops, 100k x 32, EMA) and ctrl (field_nowm_ft, same trained encoder).
Crop rules, fixed before looking at outputs:
  views      the six fig_data crops (traj_*.npz): field rendered at a turntable of poses around the GT pose
  occlusion  of 600 test crops drawn with seed 1, those whose visible mask covers 25-60% of the GT silhouette,
             first one per object in draw order, six objects
  population 48 test crops drawn with seed 3 (mask >= 10% of the crop), 8 chains each, joint and control, same noise

Run: CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python fig_data2.py
Outputs: /media/kavinder/hdd3/larc_figs/data/{views,occlusion,population}.npz
"""
import os
import sys
import types

for _m in ("bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.training import checkpoints
from jaxlie import SE3
from scipy.spatial.transform import Rotation

from liepose.dist import NormalSE3
from liepose.exp.cmp.sp_field_dit import FtFieldDenoiser
from liepose.exp.cmp.tless_dit import mask_fg
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.tless_field import TlessGeometry, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.exp.testbed import Testbed
from liepose.metrics import se3 as se3_metrics
from liepose.noise.ve import PowerNoiseSchedule

OUT = "/media/kavinder/hdd3/larc_figs/data"
DD = "/media/kavinder/hdd2/liepose_data"
AB = "/media/kavinder/hdd3/larp_abl"
lm = se3_metrics
NS = 8


def wxyz_to_mat(q):
  return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()


def restore(path):
  c = ocp.PyTreeCheckpointer()
  m = c.metadata(path)
  tree = m.item_metadata.tree if hasattr(m, "item_metadata") else m
  sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  st = c.restore(path, restore_args=jax.tree_util.tree_map(lambda x: ocp.ArrayRestoreArgs(sharding=sh), tree))
  return st["params_ema"]


meta = np.load(f"{DD}/tless_tokens/test_meta.npz")
sh = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
assert (sh["index"] == meta["index"]).all()
images, masks = sh["image"], sh["mask"]
kmeta = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
obj_all = meta["object_id"]
R_all = wxyz_to_mat(meta["rotation_allo"]).astype(np.float32)
t_all = meta["translation_scaled"].astype(np.float32)
import json
info = json.load(open(f"{DD}/bop_datasets/tless/models_cad/models_info.json"))


def sym_mats(o):
  S = [np.eye(3)] + [np.asarray(s, np.float64).reshape(4, 4)[:3, :3] for s in info[str(o)].get("symmetries_discrete", [])]
  return np.stack(S)


def rot_err(Rgt, Rm, o):
  """symmetry-aware rotation error (deg): continuous symmetry -> angle between the symmetry axes (as figs.rot_err)"""
  v = info[str(o)]
  if "symmetries_continuous" in v:
    ax = np.asarray(v["symmetries_continuous"][0]["axis"], float)
    a1 = Rgt @ ax; a2 = np.einsum("...ij,j->...i", Rm, ax)
    e = np.degrees(np.arccos(np.clip(a2 @ a1, -1, 1)))
    return np.minimum(e, 180 - e) if v.get("symmetries_discrete") else e
  e = []
  for S in sym_mats(o):
    M = np.einsum("ij,...jk->...ik", (Rgt @ S).T, Rm)
    e.append(np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1))))
  return np.min(np.stack(e), 0)


field, _ = load_tless_field(f"{AB}/probe/tl_field_v1")
pj = restore(f"{AB}/a1_ckpt/checkpoint_100001")
fparams = {"params": pj["field"]}
pj = {"params": {k: v for k, v in pj.items() if k != "field"}}
pc = {"params": restore(checkpoints.latest_checkpoint(
  f"{AB}/experiments/tless-field/train_steps=100000+seed=42+den_mode=field_nowm_ft+field=tl_field_v1/ckpt"))}
m_joint = FtFieldDenoiser(mode="field_full_lat", dim=256, depth=4, use_k=False)
m_ctrl = FtFieldDenoiser(mode="field_nowm", dim=256, depth=4, use_k=False)
enc_joint = jax.jit(lambda im: m_joint.apply(pj, im, method=FtFieldDenoiser.encode))
enc_ctrl = jax.jit(lambda im: m_ctrl.apply(pc, im, method=FtFieldDenoiser.encode))


@jax.jit
def render(obj, Rs, ts, k, z):
  n = Rs.shape[0]
  return field.apply(fparams, jnp.broadcast_to(obj, (n,)), Rs, ts, jnp.broadcast_to(k, (n, 10)),
                     jnp.broadcast_to(app_stats(z), (n, 768)), with_opacity=True, with_latent=True)


def observe(row):
  img = images[row].astype(np.float32) * masks[row][..., None]
  w = FG_FLOOR + np.asarray(mask_fg(jnp.asarray(masks[row].astype(np.float32))[None]))[0]
  return img, w


# ------------------------------------------------------------------ 1. novel views (turntable about the camera-facing pose)
if not os.path.exists(f"{OUT}/views.npz"):
  V = {}
  az = np.radians(np.arange(0, 360, 45)); el = np.radians([-30, 0, 30])
  for ci in range(6):
    d = np.load(f"{OUT}/traj_{ci}.npz")
    o, k = int(d["obj"]), jnp.asarray(d["k"])
    Rs = np.stack([Rotation.from_euler("x", e).as_matrix() @ Rotation.from_euler("y", a).as_matrix() @ d["R_gt"]
                   for e in el for a in az]).astype(np.float32)
    Ts = np.repeat(d["t_gt"][None], len(Rs), 0)
    zh, op, _ = render(o, jnp.asarray(Rs), jnp.asarray(Ts), k, jnp.asarray(d["z_joint"]))
    V[f"zh_{ci}"], V[f"op_{ci}"], V[f"R_{ci}"] = np.asarray(zh, np.float16), np.asarray(op), Rs
  np.savez(f"{OUT}/views.npz", az=az, el=el, **V)
  print("views done", flush=True)

# ------------------------------------------------------------------ 2. occlusion: amodal render, residual, WM weights
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
from bop_toolkit_lib import inout

geo = TlessGeometry()
_mesh = {}


def silhouette(o, R_allo, t_s, k, res=224, ss=2):
  """projected CAD silhouette in the crop (crop selection and display only; LaRC never uses the CAD model)"""
  if o not in _mesh:
    m = inout.load_ply(f"{DD}/bop_datasets/tless/models_eval/obj_{o:06d}.ply")
    _mesh[o] = (m["pts"] / 1000.0, np.asarray(m["faces"]).astype(int))
  v, f = _mesh[o]
  R, t = geo.cam_pose(jnp.asarray(R_allo)[None], jnp.asarray(t_s)[None], jnp.asarray(k)[None])
  Pc = v @ np.asarray(R[0]).T + np.asarray(t[0])
  u = Pc[:, :2] / Pc[:, 2:] * k[6:8] + k[8:10]
  u = (u - k[1:3]) * k[0] * ss
  im = Image.new("L", (res * ss, res * ss), 0)
  dr = ImageDraw.Draw(im)
  for tri in f:
    dr.polygon([tuple(u[i]) for i in tri], fill=255)
  return zoom(np.asarray(im) > 0, 1 / ss, order=0)


if not os.path.exists(f"{OUT}/occlusion.npz"):
  rng = np.random.default_rng(1)
  cand = rng.choice(len(obj_all), 600, replace=False)
  picked, seen = [], set()
  for r in cand:
    o = int(obj_all[r])
    if o in seen:
      continue
    sil = silhouette(o, R_all[r], t_all[r], kmeta[r])
    vf = (masks[r] > 0).sum() / max(sil.sum(), 1)
    if 0.25 <= vf <= 0.60 and sil.sum() > 2000:
      picked.append((int(r), float(vf), sil)); seen.add(o)
    if len(picked) == 6:
      break
  Oc = {"rows": np.array([p[0] for p in picked]), "vf": np.array([p[1] for p in picked])}
  for j, (r, vf, sil) in enumerate(picked):
    o, k = int(obj_all[r]), jnp.asarray(kmeta[r])
    img, w = observe(r)
    raw = images[r].astype(np.float32)
    for nm, im in (("masked", img), ("raw", raw)):  # encoder input with and without the annotation mask
      z = enc_joint(jnp.asarray(im)[None])[0][0]
      zh, op, _ = render(o, jnp.asarray(R_all[r])[None], jnp.asarray(t_all[r])[None], k, z)
      r2 = np.asarray(((zh[0] - z) ** 2).mean(-1))
      k2 = np.median(r2) + 1e-8
      Oc[f"{nm}_res_{j}"] = r2
      Oc[f"{nm}_op_{j}"] = np.asarray(op[0])
      Oc[f"{nm}_wm_{j}"] = FG_FLOOR + np.clip(np.asarray(op[0]), 0, 1) * k2 / (k2 + r2)  # = tless_field_dit._weights
    Oc[f"img_{j}"], Oc[f"mask_{j}"], Oc[f"sil_{j}"], Oc[f"w_{j}"] = images[r], masks[r], sil, w
    Oc[f"obj_{j}"] = o
  np.savez(f"{OUT}/occlusion.npz", **Oc)
  print("occlusion done", Oc["rows"], np.round(Oc["vf"], 2), flush=True)

# ------------------------------------------------------------------ 3. population: 48 crops, 8 chains, joint vs control
ns = PowerNoiseSchedule(1e-8, 1.0, 100, 3)
fake = types.SimpleNamespace(Lie=SE3, LieDist=NormalSE3, lie_metrics=lm, noise_schedule=ns,
                             a=types.SimpleNamespace(learn_noise=True, repr_type="tan"),
                             opt=types.SimpleNamespace(beta_skip=True, eta=1.0))
p_fn = jax.vmap(Testbed.create_p_sample_score_fn(fake))
cur = (np.linspace(100, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
seq = list(zip(cur, cur[1:] + [0]))
poses = jax.jit(jax.vmap(lambda r: (lm.as_lie(r).rotation().as_matrix(), lm.as_lie(r).translation())))
prior = jax.jit(jax.vmap(lambda kk: lm.as_repr(SE3.sample_uniform(kk), "tan")))


def make_step(model, prm):
  @jax.jit
  def step(key, rt, t, tp, obj, z, w, k, ze, zcmp):
    """z: the model's tokens; zcmp: LaRC's comparison tokens (render and the reported energy, same for both models)"""
    key, kk = jax.random.split(key)
    Rg, tg = poses(rt)
    zh, opa, h = render(obj, Rg, tg, k, zcmp)
    mu = model.apply(prm, rt, jnp.full((NS, 1), t, jnp.float32), jnp.broadcast_to(k, (NS, 10)),
                     jnp.broadcast_to(z, (NS, 64, 384)), jnp.broadcast_to(w, (NS, 64)), zh, opa, h, None,
                     jnp.broadcast_to(ze, (NS,) + ze.shape))
    _, rt = p_fn(jax.random.split(kk, NS), mu, rt, jnp.full((NS,), t), jnp.full((NS,), tp))
    e = weighted_energy(zh, jnp.broadcast_to(zcmp, zh.shape), jnp.broadcast_to(w, (NS, 64)))
    return key, rt, e
  return step


step_j, step_c = make_step(m_joint, pj), make_step(m_ctrl, pc)
P = f"{OUT}/population.npz"
if not os.path.exists(P):
  rng = np.random.default_rng(3)
  mfrac = masks.reshape(len(masks), -1).mean(1)
  pool = np.where(mfrac >= 0.10)[0]
  prow = rng.choice(pool, 48, replace=False)
  out = {"rows": prow}
  for nm in ("joint", "ctrl"):
    out[f"{nm}_rerr"], out[f"{nm}_terr"], out[f"{nm}_E"], out[f"{nm}_R"], out[f"{nm}_T"] = [], [], [], [], []
  for i, r in enumerate(prow):
    o, k = int(obj_all[r]), jnp.asarray(kmeta[r])
    img, w = observe(r)
    key0 = jax.random.PRNGKey(3000 + i)
    k1, k2 = jax.random.split(key0)
    rt0 = prior(jax.random.split(k1, NS))
    zj, zej = [x[0] for x in enc_joint(jnp.asarray(img)[None])]
    zc, zec = [x[0] for x in enc_ctrl(jnp.asarray(img)[None])]
    for nm, step, z, ze in (("joint", step_j, zj, zej), ("ctrl", step_c, zc, zec)):
      key, rt = k2, rt0
      Rs, Ts, Es = [], [], []
      for t, tp in seq:
        key, rt, e = step(key, rt, jnp.int32(t), jnp.int32(tp), o, z, jnp.asarray(w), k, ze, zj)
        R_, T_ = poses(rt)
        Rs.append(np.asarray(R_)); Ts.append(np.asarray(T_)); Es.append(np.asarray(e))
      Rs, Ts = np.stack(Rs), np.stack(Ts)
      out[f"{nm}_rerr"].append(rot_err(R_all[r], Rs, o)); out[f"{nm}_terr"].append(np.linalg.norm(Ts - t_all[r], axis=-1))
      out[f"{nm}_E"].append(np.stack(Es)); out[f"{nm}_R"].append(Rs[-1]); out[f"{nm}_T"].append(Ts[-1])
    print("population", i, "joint", np.round(out["joint_rerr"][-1][-1], 1), "ctrl", np.round(out["ctrl_rerr"][-1][-1], 1), flush=True)
  np.savez(P, **{k: np.stack(v) if isinstance(v, list) else v for k, v in out.items()},
           obj=obj_all[prow], R_gt=R_all[prow], t_gt=t_all[prow], k=kmeta[prow])
print("done", flush=True)
