"""Real inference for the paper's figures (read-only use of the trained checkpoints; no training, no test-set tuning).

Models (T-LESS, GT crops, 100k x 32, EMA weights):
  joint    LaRC, the paper's method: field_full_lat_ft + joint (EMA comparison encoder, field trained jointly)
  ctrl     control with the same trained encoder: field_nowm_ft (no world-model inputs)
  frozen   LaRC with frozen DINOv2 and the frozen field tl_field_v1 (ablation; comparison-space figure only)

Crop rule, fixed before looking at any output: for each object in OBJS, the first test crop in cache order whose visible
mask covers at least 25% of the crop (the explainer video's rule). Discrimination crops: 400 test crops drawn with a fixed seed.

Run: CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=LARp/src python fig_data.py
Outputs: /media/kavinder/hdd3/larc_figs/data/*.npz
"""
import json
import os
import sys
import types

for _m in ("bop_toolkit_lib", "bop_toolkit_lib.config", "bop_toolkit_lib.inout", "bop_toolkit_lib.misc",
           "bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))
for _s in ("config", "inout", "misc"):
  setattr(sys.modules["bop_toolkit_lib"], _s, sys.modules[f"bop_toolkit_lib.{_s}"])

import flax
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
from liepose.exp.jepa_probe.tless_field import TlessField, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.exp.testbed import Testbed
from liepose.metrics import se3 as se3_metrics
from liepose.noise.ve import PowerNoiseSchedule

OUT = "/media/kavinder/hdd3/larc_figs/data"
DD = "/media/kavinder/hdd2/liepose_data"
AB = "/media/kavinder/hdd3/larp_abl"
VC = "/media/kavinder/hdd3/larc_video/ckpt"
OBJS = (5, 8, 18, 21, 23, 27)
NS = 16
os.makedirs(OUT, exist_ok=True)
lm = se3_metrics


def wxyz_to_mat(q):
  return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()


def restore(path):
  c = ocp.PyTreeCheckpointer()
  m = c.metadata(path)
  tree = m.item_metadata.tree if hasattr(m, "item_metadata") else m
  sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  st = c.restore(path, restore_args=jax.tree_util.tree_map(lambda x: ocp.ArrayRestoreArgs(sharding=sh), tree))
  print(path, "step", st.get("step"))
  return st["params_ema"]


# ------------------------------------------------------------------ data
meta = np.load(f"{DD}/tless_tokens/test_meta.npz")
sh = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
assert (sh["index"] == meta["index"]).all()
images, masks = sh["image"], sh["mask"]
kmeta = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
ztok = np.load(f"{DD}/tless_tokens/test_tokens.npy", mmap_mode="r")
zfg = np.load(f"{DD}/tless_tokens/test_fg.npy", mmap_mode="r")
obj_all = meta["object_id"]
R_all = wxyz_to_mat(meta["rotation_allo"]).astype(np.float32)
t_all = meta["translation_scaled"].astype(np.float32)
mfrac = masks.reshape(len(masks), -1).mean(1)
rows = [int(np.where((obj_all == o) & (mfrac >= 0.25))[0][0]) for o in OBJS]
print("rows", rows)
info = json.load(open(f"{DD}/bop_datasets/tless/models_cad/models_info.json"))


def sym_mats(o):
  S = [np.eye(3)]
  for s in info[str(o)].get("symmetries_discrete", []):
    S.append(np.asarray(s, np.float64).reshape(4, 4)[:3, :3])
  return np.stack(S)


def rot_err(Rgt, Rm, o):
  """symmetry-aware geodesic rotation error (deg); Rm (..., 3, 3)"""
  e = []
  for S in sym_mats(o):
    M = np.einsum("ij,...jk->...ik", (Rgt @ S).T, Rm)
    e.append(np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1))))
  return np.min(np.stack(e), 0)


# ------------------------------------------------------------------ models
field, fparams_frozen = load_tless_field(f"{AB}/probe/tl_field_v1")
pj = restore(f"{AB}/a1_ckpt/checkpoint_100001")
fparams_joint = {"params": pj["field"]}
pj = {"params": {k: v for k, v in pj.items() if k != "field"}}
pc = {"params": restore(checkpoints.latest_checkpoint(
  f"{AB}/experiments/tless-field/train_steps=100000+seed=42+den_mode=field_nowm_ft+field=tl_field_v1/ckpt"))}
m_joint = FtFieldDenoiser(mode="field_full_lat", dim=256, depth=4, use_k=False)
m_ctrl = FtFieldDenoiser(mode="field_nowm", dim=256, depth=4, use_k=False)
enc_joint = jax.jit(lambda im: m_joint.apply(pj, im, method=FtFieldDenoiser.encode))
enc_ctrl = jax.jit(lambda im: m_ctrl.apply(pc, im, method=FtFieldDenoiser.encode))


def make_render(fp):
  @jax.jit
  def render(obj, Rs, ts, k, z):
    n = Rs.shape[0]
    return field.apply(fp, jnp.broadcast_to(obj, (n,)), Rs, ts, jnp.broadcast_to(k, (n, 10)),
                       jnp.broadcast_to(app_stats(z), (n, 768)), with_opacity=True, with_latent=True)
  return render


render_j, render_f = make_render(fparams_joint), make_render(fparams_frozen)


def energies(render, obj, Rs, ts, k, z, w, bs=24):
  out = []
  for i in range(0, len(Rs), bs):
    zh = render(obj, jnp.asarray(Rs[i:i + bs]), jnp.asarray(ts[i:i + bs]), k, z)[0]
    out.append(np.asarray(weighted_energy(zh, z[None], w[None])))
  return np.concatenate(out)


def observe(row):
  img = images[row].astype(np.float32) * masks[row][..., None]
  m = masks[row].astype(np.float32)
  w = FG_FLOOR + np.asarray(mask_fg(jnp.asarray(m)[None]))[0]
  return img, w


# ------------------------------------------------------------------ 1. sampler runs: joint vs control, same priors and noise
ns = PowerNoiseSchedule(1e-8, 1.0, 100, 3)
fake = types.SimpleNamespace(Lie=SE3, LieDist=NormalSE3, lie_metrics=lm, noise_schedule=ns,
                             a=types.SimpleNamespace(learn_noise=True, repr_type="tan"),
                             opt=types.SimpleNamespace(beta_skip=True, eta=1.0))
p_fn = jax.vmap(Testbed.create_p_sample_score_fn(fake))
cur = (np.linspace(100, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
seq = list(zip(cur, cur[1:] + [0]))
poses = jax.jit(jax.vmap(lambda r: (lm.as_lie(r).rotation().as_matrix(), lm.as_lie(r).translation())))
prior = jax.jit(jax.vmap(lambda kk: lm.as_repr(SE3.sample_uniform(kk), "tan")))


def make_step(model, prm, render):
  @jax.jit
  def step(key, rt, t, tp, obj, z, w, k, ze):
    key, kk = jax.random.split(key)
    Rg, tg = poses(rt)
    zh, opa, h = render(obj, Rg, tg, k, z)
    mu = model.apply(prm, rt, jnp.full((NS, 1), t, jnp.float32), jnp.broadcast_to(k, (NS, 10)),
                     jnp.broadcast_to(z, (NS, 64, 384)), jnp.broadcast_to(w, (NS, 64)), zh, opa, h, None,
                     jnp.broadcast_to(ze, (NS,) + ze.shape))
    _, rt = p_fn(jax.random.split(kk, NS), mu, rt, jnp.full((NS,), t), jnp.full((NS,), tp))
    return key, rt, zh, opa
  return step


step_j, step_c = make_step(m_joint, pj, render_j), make_step(m_ctrl, pc, render_j)
traj = {}
for ci, row in enumerate(rows):
  if os.path.exists(f"{OUT}/traj_{ci}.npz"):
    continue
  o = int(obj_all[row])
  img, w = observe(row)
  k = jnp.asarray(kmeta[row])
  zj, zej = [x[0] for x in enc_joint(jnp.asarray(img)[None])]
  zc, zec = [x[0] for x in enc_ctrl(jnp.asarray(img)[None])]
  key0 = jax.random.PRNGKey(2026 + ci)
  k1, k2 = jax.random.split(key0)
  rt0 = prior(jax.random.split(k1, NS))
  d = dict(row=row, obj=o, img=images[row], mask=masks[row], k=kmeta[row], R_gt=R_all[row], t_gt=t_all[row], w=w,
           z_joint=np.asarray(zj), z_frozen=np.asarray(ztok[row], np.float32), z_ctrl=np.asarray(zc))
  for name, step, z, ze in (("joint", step_j, zj, zej), ("ctrl", step_c, zc, zec)):
    key, rt = k2, rt0
    rts, zh0, op0 = [np.asarray(rt)], [], []
    for t, tp in seq:
      key, rt, zh, opa = step(key, rt, jnp.int32(t), jnp.int32(tp), o, z, jnp.asarray(w), k, ze)
      rts.append(np.asarray(rt))
      zh0.append(np.asarray(zh[0], np.float16)); op0.append(np.asarray(opa[0]))
    rts = np.stack(rts)
    Rt, Tt = map(np.stack, zip(*[[np.asarray(x) for x in poses(jnp.asarray(r))] for r in rts]))
    d[f"{name}_R"], d[f"{name}_T"] = Rt, Tt
    d[f"{name}_rerr"] = rot_err(R_all[row], Rt, o)
    d[f"{name}_terr"] = np.linalg.norm(Tt - t_all[row], axis=-1)
    d[f"{name}_E"] = np.stack([energies(render_j, o, Rt[i], Tt[i], k, zj, jnp.asarray(w)) for i in range(0, 101, 5)])
    d[f"{name}_zh0"], d[f"{name}_op0"] = np.stack(zh0), np.stack(op0)
    print(f"obj {o} row {row} {name}: final rot err {np.round(d[f'{name}_rerr'][-1], 1)}", flush=True)
  # renders at the GT pose (joint field + joint tokens; frozen field + frozen tokens)
  for nm, rend, z in (("joint", render_j, zj), ("frozen", render_f, jnp.asarray(d["z_frozen"]))):
    zh, opa, h = rend(o, jnp.asarray(R_all[row])[None], jnp.asarray(t_all[row])[None], k, z)
    d[f"gt_zh_{nm}"], d[f"gt_op_{nm}"], d[f"gt_h_{nm}"] = np.asarray(zh[0]), np.asarray(opa[0]), np.asarray(h[0])
  np.savez(f"{OUT}/traj_{ci}.npz", **d)

# ------------------------------------------------------------------ 2. energy landscapes (crop 0, object 5: one 180 deg symmetry)
d = np.load(f"{OUT}/traj_0.npz")
o, row, k = int(d["obj"]), int(d["row"]), jnp.asarray(d["k"])
S = sym_mats(o)[1]
s_axis = Rotation.from_matrix(S).as_rotvec(); s_axis /= np.linalg.norm(s_axis)
u = np.cross(s_axis, [1.0, 0, 0]); u = u if np.linalg.norm(u) > 0.3 else np.cross(s_axis, [0, 1.0, 0]); u /= np.linalg.norm(u)
th, ph = np.radians(np.linspace(-180, 180, 73)), np.radians(np.linspace(-90, 90, 37))
Rs = np.stack([d["R_gt"] @ Rotation.from_rotvec(a * s_axis).as_matrix() @ Rotation.from_rotvec(b * u).as_matrix()
               for b in ph for a in th]).astype(np.float32)
Ts = np.repeat(d["t_gt"][None], len(Rs), 0)
land = dict(theta=th, phi=ph)
for nm, rend, z in (("joint", render_j, d["z_joint"]), ("frozen", render_f, d["z_frozen"])):
  land[nm] = energies(rend, o, Rs, Ts, k, jnp.asarray(z), jnp.asarray(d["w"])).reshape(len(ph), len(th))
np.savez(f"{OUT}/landscape.npz", **land)
print("landscape done", flush=True)

# ------------------------------------------------------------------ 3. discrimination: E(g* Exp(delta u)) / E(g*) on 400 crops
rng = np.random.default_rng(0)
drows = rng.choice(len(obj_all), 400, replace=False)
deltas = np.array([0, 1, 2, 5, 10, 20, 45, 90, 180], np.float32)
n_ax = 8
disc = {"rows": drows, "deltas": deltas, "joint": [], "frozen": []}
for i, r in enumerate(drows):
  o = int(obj_all[r]); k = jnp.asarray(kmeta[r])
  img, w = observe(r)
  zj = enc_joint(jnp.asarray(img)[None])[0][0]
  ax = rng.normal(size=(n_ax, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
  Rs = np.stack([R_all[r] @ Rotation.from_rotvec(np.radians(dl) * a).as_matrix() for dl in deltas for a in ax]).astype(np.float32)
  Ts = np.repeat(t_all[r][None], len(Rs), 0)
  re = rot_err(R_all[r], Rs, o).reshape(len(deltas), n_ax)  # symmetry-aware distance of each perturbed pose
  for nm, rend, z in (("joint", render_j, zj), ("frozen", render_f, jnp.asarray(np.asarray(ztok[r], np.float32)))):
    E = energies(rend, o, Rs, Ts, k, z, jnp.asarray(w)).reshape(len(deltas), n_ax)
    disc[nm].append(np.stack([E, re]))
  if i % 50 == 0:
    print("disc", i, flush=True)
np.savez(f"{OUT}/disc.npz", rows=drows, deltas=deltas, joint=np.stack(disc["joint"]), frozen=np.stack(disc["frozen"]),
         obj=obj_all[drows])

# ------------------------------------------------------------------ 4. field density (never supervised), joint and frozen
n = 48
g = (np.arange(n) + 0.5) / n * 2 - 1
G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(1, -1, 3).astype(np.float32)
dens = {}
for o in OBJS:
  for nm, fp in (("joint", fparams_joint), ("frozen", fparams_frozen)):
    sig, _ = field.apply(fp, jnp.asarray(G), jnp.array([o - 1]), method=TlessField.point)
    dens[f"{nm}_{o}"] = np.asarray(sig[0]).reshape(n, n, n)
  dens[f"r_{o}"] = float(field.radii[o - 1])
np.savez(f"{OUT}/density.npz", grid=g, **dens)
print("done", flush=True)
