"""Real data for the explainer video (read-only use of the paper's code, data and checkpoints; CPU only).

Run with the existing liepose env (not modified):
  JAX_PLATFORMS=cpu PYTHONPATH=/home/kavinder/ARNABI_ARSH/LARp/src \
    /home/kavinder/miniconda3/envs/liepose/bin/python video/tools/real_data.py

Outputs (npz / png) to /media/kavinder/hdd3/larc_video/data:
  crop.png, mask.png         the chosen T-LESS test crop (fixed rule below) and its visible mask
  tokens.npz                 observed tokens z, weights w, rendered z_hat / opacity / latent at the GT pose and at a wrong
                             pose, PCA colours (one basis), per-token squared distance
  landscape.npz              E(g; z) on a 2-D slice of SO(3) through the GT and its symmetric copy (field only)
  density.npz                the field's density and latent on a grid in the object box (never supervised)
  traj.npz                   real sampler runs of LaRC and the control (100k x 32 checkpoints, EMA weights), same priors
                             and same noise, 16 samples, every step; energies along the way; GT only to measure errors
Crop rule (fixed before looking at any output): object 5 (one discrete 180 deg symmetry), the first test crop in cache
order whose visible mask covers at least 25% of the crop.
"""
import json
import os
import sys
import types

for _m in ('bop_toolkit_lib', 'bop_toolkit_lib.config', 'bop_toolkit_lib.inout', 'bop_toolkit_lib.misc',
           'bop_toolkit_lib.renderer', 'bop_toolkit_lib.pose_error'):
  sys.modules.setdefault(_m, types.ModuleType(_m))  # BOP toolkit: imported by the evaluator only, unused here
sys.modules['bop_toolkit_lib'].config = sys.modules['bop_toolkit_lib.config']
sys.modules['bop_toolkit_lib'].inout = sys.modules['bop_toolkit_lib.inout']
sys.modules['bop_toolkit_lib'].misc = sys.modules['bop_toolkit_lib.misc']

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from jaxlie import SE3, SO3
from PIL import Image
from scipy.spatial.transform import Rotation

from liepose.dist import NormalSE3
from liepose.exp.cmp.sp_field_dit import FieldDenoiser
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.tless_field import TlessField, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.exp.testbed import Testbed
from liepose.metrics import se3 as se3_metrics
from liepose.noise.ve import PowerNoiseSchedule

OUT = "/media/kavinder/hdd3/larc_video/data"
CK = "/media/kavinder/hdd3/larc_video/ckpt"
DD = "/media/kavinder/hdd2/liepose_data"
TL = f"{DD}/bop_datasets/tless"
OBJ = 5
os.makedirs(OUT, exist_ok=True)


def wxyz_to_mat(q):
  return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()


# ---------------------------------------------------------------- crop
meta = np.load(f"{DD}/tless_tokens/test_meta.npz")
sh = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
assert (sh["index"] == meta["index"]).all()
obj_all, masks = meta["object_id"], sh["mask"]
cand = [i for i in range(len(obj_all)) if obj_all[i] == OBJ and masks[i].mean() >= 0.25]
row = cand[0]
print("row", row, "index", int(meta["index"][row]), "mask frac", masks[row].mean())
img = sh["image"][row]
Image.fromarray(img).save(f"{OUT}/crop.png")
Image.fromarray((masks[row] * 255).astype(np.uint8)).save(f"{OUT}/mask.png")
Image.fromarray((img * masks[row][..., None]).astype(np.uint8)).save(f"{OUT}/crop_masked.png")

z = np.load(f"{DD}/tless_tokens/test_tokens.npy", mmap_mode="r")[row].astype(np.float32)
fg = np.load(f"{DD}/tless_tokens/test_fg.npy", mmap_mode="r")[row].astype(np.float32)
k = np.load(f"{DD}/tless_tokens/test_crop_meta.npy")[row].astype(np.float32)
R_gt = wxyz_to_mat(meta["rotation_allo"][row]).astype(np.float32)
t_gt = meta["translation_scaled"][row].astype(np.float32)
w = FG_FLOOR + fg

info = json.load(open(f"{TL}/models_cad/models_info.json"))[str(OBJ)]
S = np.asarray(info["symmetries_discrete"][0], np.float64).reshape(4, 4)[:3, :3]
s_axis = Rotation.from_matrix(S).as_rotvec()
print("symmetry rotvec", s_axis, np.degrees(np.linalg.norm(s_axis)))
s_axis = s_axis / np.linalg.norm(s_axis)

# ---------------------------------------------------------------- field
field, fparams = load_tless_field(f"{CK}/tl_field_v1")
app = app_stats(jnp.asarray(z))[None]


@jax.jit
def render(Rs, ts):
  n = Rs.shape[0]
  return field.apply(fparams, jnp.full((n,), OBJ), Rs, ts, jnp.broadcast_to(jnp.asarray(k), (n, 10)),
                     jnp.broadcast_to(app, (n, 768)), with_latent=True)


def energies(Rs, ts, bs=64):
  out = []
  for i in range(0, len(Rs), bs):
    zh, _, _ = render(jnp.asarray(Rs[i:i + bs], jnp.float32), jnp.asarray(ts[i:i + bs], jnp.float32))
    out.append(np.asarray(weighted_energy(zh, jnp.asarray(z)[None], jnp.asarray(w)[None])))
  return np.concatenate(out)


zh_gt, op_gt, h_gt = [np.asarray(x[0]) for x in render(jnp.asarray(R_gt)[None], jnp.asarray(t_gt)[None])]
R_wrong = (R_gt @ Rotation.from_rotvec(np.radians(40) * np.array([0.6, 0.8, 0.0])).as_matrix()).astype(np.float32)
zh_wr, op_wr, h_wr = [np.asarray(x[0]) for x in render(jnp.asarray(R_wrong)[None], jnp.asarray(t_gt)[None])]
E_gt = float(energies(R_gt[None], t_gt[None])[0])
E_wr = float(energies(R_wrong[None], t_gt[None])[0])
print("E gt", E_gt, "E wrong", E_wr)

# PCA basis on observed + rendered tokens (one basis for every token picture)
X = np.concatenate([z, zh_gt, zh_wr], 0)
mu = X.mean(0)
U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
P = Vt[:3]
proj = lambda a: (a - mu) @ P.T
allp = proj(X)
lo, hi = np.percentile(allp, 2, 0), np.percentile(allp, 98, 0)
col = lambda a: np.clip((proj(a) - lo) / (hi - lo), 0, 1)
np.savez(f"{OUT}/tokens.npz", z=z, w=w, fg=fg, k=k, R_gt=R_gt, t_gt=t_gt, R_wrong=R_wrong,
         zh_gt=zh_gt, op_gt=op_gt, h_gt=h_gt, zh_wr=zh_wr, op_wr=op_wr, h_wr=h_wr,
         col_obs=col(z), col_gt=col(zh_gt), col_wr=col(zh_wr),
         d_gt=((zh_gt - z) ** 2).mean(-1), d_wr=((zh_wr - z) ** 2).mean(-1), E_gt=E_gt, E_wr=E_wr,
         row=row, index=int(meta["index"][row]), S=S)

# ---------------------------------------------------------------- energy landscape on a slice of SO(3)
u = np.cross(s_axis, [1.0, 0, 0])
if np.linalg.norm(u) < 0.3:
  u = np.cross(s_axis, [0, 1.0, 0])
u /= np.linalg.norm(u)
th = np.radians(np.linspace(-180, 180, 73))
ph = np.radians(np.linspace(-90, 90, 37))
Rs = np.stack([R_gt @ Rotation.from_rotvec(a * s_axis).as_matrix() @ Rotation.from_rotvec(b * u).as_matrix()
               for b in ph for a in th]).astype(np.float32)
E = energies(Rs, np.repeat(t_gt[None], len(Rs), 0)).reshape(len(ph), len(th))
# translation slice (depth and x, crop-relative parametrisation), rotation at GT
dx = np.linspace(-1.0, 1.0, 41)
dz = np.linspace(-1.0, 1.0, 41)
ts = np.stack([t_gt + np.array([a, 0, b]) for b in dz for a in dx]).astype(np.float32)
Et = energies(np.repeat(R_gt[None], len(ts), 0), ts).reshape(len(dz), len(dx))
np.savez(f"{OUT}/landscape.npz", theta=th, phi=ph, E=E, s_axis=s_axis, u_axis=u, dx=dx, dz=dz, Et=Et, E_gt=E_gt)
print("landscape", E.min(), E.max(), "at gt", E[len(ph) // 2, len(th) // 2], "at sym", E[len(ph) // 2, -1])

# ---------------------------------------------------------------- density of the field in the object box
n = 40
g = (np.arange(n) + 0.5) / n * 2 - 1
G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(1, -1, 3).astype(np.float32)
sig, hh = field.apply(fparams, jnp.asarray(G), jnp.array([OBJ - 1]), method=TlessField.point)
sig, hh = np.asarray(sig[0]), np.asarray(hh[0])
r_o = float(field.radii[OBJ - 1])
np.savez(f"{OUT}/density.npz", grid=G[0], sigma=sig, h_pca=(hh - hh.mean(0)) @ np.linalg.svd(hh - hh.mean(0), full_matrices=False)[2][:3].T,
         r_o=r_o)
print("density", np.percentile(sig, [50, 90, 99, 100]), "r_o", r_o)

# ---------------------------------------------------------------- real sampler, LaRC vs control
ns = PowerNoiseSchedule(1e-8, 1.0, 100, 3)
fake = types.SimpleNamespace(Lie=SE3, LieDist=NormalSE3, lie_metrics=se3_metrics, noise_schedule=ns,
                             a=types.SimpleNamespace(learn_noise=True, repr_type="tan"),
                             opt=types.SimpleNamespace(beta_skip=True, eta=1.0))
p_fn = jax.vmap(Testbed.create_p_sample_score_fn(fake))
lm = se3_metrics
cur = (np.linspace(100, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
seq = list(zip(cur, cur[1:] + [0]))
NS = 16


def load_den(mode):
  d = f"{CK}/tless-field/train_steps=100000+seed=42+den_mode={mode}+field=tl_field_v1/ckpt"
  path = checkpoints.latest_checkpoint(d)
  import orbax.checkpoint as ocp
  c = ocp.PyTreeCheckpointer()
  m = c.metadata(path)
  tree = m.item_metadata.tree if hasattr(m, "item_metadata") else m
  sh_ = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  st = c.restore(path, restore_args=jax.tree_util.tree_map(lambda x: ocp.ArrayRestoreArgs(sharding=sh_), tree))
  print(mode, path, "step", st.get("step"), list(st.keys()))
  return FieldDenoiser(mode=mode, dim=256, depth=4, use_k=False), {"params": st["params_ema"]}


poses = jax.jit(jax.vmap(lambda r: (lm.as_lie(r).rotation().as_matrix(), lm.as_lie(r).translation())))
prior = jax.jit(jax.vmap(lambda kk: lm.as_repr(SE3.sample_uniform(kk), "tan")))
key0 = jax.random.PRNGKey(2026)
k1, k2 = jax.random.split(key0)
rt0 = prior(jax.random.split(k1, NS))
zb = jnp.broadcast_to(jnp.asarray(z), (NS, 64, 384))
wb = jnp.broadcast_to(jnp.asarray(w), (NS, 64))
kb = jnp.broadcast_to(jnp.asarray(k), (NS, 10))
res = {}
for mode in ("field_full_lat", "field_nowm"):
  model, prm = load_den(mode)

  @jax.jit
  def step(key, rt, t, tp):
    key, kk = jax.random.split(key)
    Rg, tg = poses(rt)
    z_hat, opa, h = render(Rg, tg)
    mu = model.apply(prm, rt, jnp.full((NS, 1), t, jnp.float32), kb, zb, wb, z_hat, opa, h, None, None)
    _, rt = p_fn(jax.random.split(kk, NS), mu, rt, jnp.full((NS,), t), jnp.full((NS,), tp))
    return key, rt, z_hat

  key, rt = k2, rt0
  traj, zh0 = [np.asarray(rt)], []
  for t, tp in seq:
    key, rt, zhat = step(key, rt, jnp.int32(t), jnp.int32(tp))
    traj.append(np.asarray(rt))
    zh0.append(np.asarray(zhat[0], np.float16))
  traj = np.stack(traj)  # (101, NS, 6)
  Rt, Tt = [], []
  for i in range(traj.shape[0]):
    a, b = poses(jnp.asarray(traj[i]))
    Rt.append(np.asarray(a)); Tt.append(np.asarray(b))
  Rt, Tt = np.stack(Rt), np.stack(Tt)
  Ef = energies(Rt[-1], Tt[-1])
  E_traj = np.stack([energies(Rt[i], Tt[i]) for i in range(0, 101, 5)])
  # symmetric rotation error to GT (diagnostic only)
  def rerr(Rm):
    e = []
    for Ss in (np.eye(3), S):
      M = np.einsum("ij,...jk->...ik", (R_gt @ Ss).T, Rm)
      e.append(np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1))))
    return np.minimum(*e)
  res[mode] = dict(rt=traj, R=Rt, T=Tt, E_final=Ef, E_traj=E_traj, rerr=rerr(Rt), terr=np.linalg.norm(Tt - t_gt, axis=-1),
                   zh0=np.stack(zh0))
  print(mode, "final rot err", np.round(res[mode]["rerr"][-1], 1), "\n  E", np.round(Ef, 3))

np.savez(f"{OUT}/traj.npz", **{f"{m}_{kk}": v for m, d in res.items() for kk, v in d.items()},
         t_seq=np.array(seq), sigma=np.asarray(ns.sqrt_alphas))
print("done")
