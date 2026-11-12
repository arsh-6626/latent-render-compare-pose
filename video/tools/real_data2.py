"""Per-ray volume-rendering quantities of the real field at the GT pose and at a wrong pose, and the decimated CAD mesh
projected into the crop at those poses and at the final samples of traj.npz (for the render-and-compare picture).
Same environment as real_data.py (liepose env, CPU, read-only). Uses the field's own geometry code (TlessGeometry)."""
import sys
import types

for _m in ("bop_toolkit_lib", "bop_toolkit_lib.config", "bop_toolkit_lib.inout", "bop_toolkit_lib.misc"):
  sys.modules.setdefault(_m, types.ModuleType(_m))

import jax.numpy as jnp
import numpy as np

from liepose.exp.jepa_probe.tless_field import TlessField, TlessGeometry, load_tless_field

OUT = "/media/kavinder/hdd3/larc_video/data"
OBJ = 5
T = np.load(f"{OUT}/tokens.npz")
M = np.load(f"{OUT}/mesh5.npz")
TR = np.load(f"{OUT}/traj.npz")
k = T["k"]
field, fparams = load_tless_field("/media/kavinder/hdd3/larc_video/ckpt/tl_field_v1")
r = float(field.radii[OBJ - 1])
geo = TlessGeometry()


def rays(R_allo, t_s):
  """replicates TlessField.__call__ up to the weights, for one pose (bin-centre samples, as at test time)"""
  R, t = geo.cam_pose(jnp.asarray(R_allo)[None], jnp.asarray(t_s)[None], jnp.asarray(k)[None])
  S, Mn = 16, 32
  cc = (jnp.arange(S) + 0.5) * (224.0 / S)
  pc = jnp.stack(jnp.meshgrid(cc, cc, indexing="xy"), -1).reshape(-1, 2)
  d = geo.ray(geo.from_crop(pc[None], jnp.asarray(k)[None]), jnp.asarray(k)[None])
  off = (jnp.arange(Mn) + 0.5) / Mn
  z = t[:, None, None, 2] + r * (2.0 * off - 1.0)
  z = jnp.broadcast_to(z, d.shape[:2] + (Mn,))
  P = d[:, :, None] * z[..., None]
  X = jnp.einsum("nji,nrmj->nrmi", R, P - t[:, None, None]) / r
  sigma, h = field.apply(fparams, X, jnp.array([OBJ - 1]), method=TlessField.point)
  dz = 2.0 * r / Mn * jnp.linalg.norm(d, axis=-1)[..., None]
  alpha = 1.0 - jnp.exp(-sigma * dz)
  Tr = jnp.cumprod(jnp.concatenate([jnp.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1] + 1e-10], -1), -1)
  w = Tr * alpha
  return dict(R_cam=np.asarray(R[0]), t_cam=np.asarray(t[0]), d=np.asarray(d[0]), z=np.asarray(z[0]), P=np.asarray(P[0]),
              X=np.asarray(X[0]), sigma=np.asarray(sigma[0]), alpha=np.asarray(alpha[0]), T=np.asarray(Tr[0]), w=np.asarray(w[0]),
              opa16=np.asarray(w[0].sum(-1)))


def project(R_allo, t_s):
  """mesh vertices (object frame, m) -> crop pixels (224 x 224) through the crop camera"""
  R, t = geo.cam_pose(jnp.asarray(R_allo)[None], jnp.asarray(t_s)[None], jnp.asarray(k)[None])
  Pc = np.asarray(M["v"]) @ np.asarray(R[0]).T + np.asarray(t[0])
  u = Pc[:, :2] / Pc[:, 2:] * k[6:8] + k[8:10]
  return (u - k[1:3]) * k[0], Pc[:, 2]


out = {}
for name, (Ra, ts) in {"gt": (T["R_gt"], T["t_gt"]), "wr": (T["R_wrong"], T["t_gt"])}.items():
  for kk, v in rays(Ra, ts).items():
    out[f"{name}_{kk}"] = v
  uv, zc = project(Ra, ts)
  out[f"{name}_uv"], out[f"{name}_zc"] = uv, zc
for mode in ("field_full_lat", "field_nowm"):
  Rf, Tf = TR[f"{mode}_R"][-1], TR[f"{mode}_T"][-1]
  out[f"{mode}_final_uv"] = np.stack([project(Rf[i], Tf[i])[0] for i in range(len(Rf))])
np.savez(f"{OUT}/rays.npz", r_o=r, **out)
print("opacity16 gt", out["gt_opa16"].reshape(16, 16).max(), "wr", out["wr_opa16"].max(), "uv range", out["gt_uv"].min(0), out["gt_uv"].max(0))
