"""Held-out regions for the world-model extrapolation tests (T-LESS). For each region: the training rows removed from the
field fit and the test crops that lie inside the region. Thresholds follow fixed rules stated below, chosen before any
held-out field is fitted; test poses are used only to place the regions where test crops exist (evaluation design).

  cone60, cone90  viewing-direction cones (as tless_field.in_holdout_cone, symmetry-aware), the same per-object directions as
                  the 30-degree test; objects with a continuous symmetry are left out (a wide cone there removes nearly all
                  their views; the elevation region covers them)
  elev            elevation of the viewing direction above the object's xy-plane (folded to |elevation| when a symmetry maps
                  z to -z): training views above the 75th percentile of the test elevations are removed
  dist            camera distance of the object: training crops farther than the 70th percentile of the test distances
  roll            in-plane rotation about the viewing axis, relative to the projected object z-axis (mod 180 deg when a
                  symmetry flips z): training views within 30 deg of the object's median test roll

Run: PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python wm_holdout_regions.py   -> /media/kavinder/hdd3/larc_figs/holdout/<name>/
"""
import json
import os
import sys
import types

for _m in ("bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from liepose.exp.jepa_probe.tless_field import TlessGeometry, in_holdout_cone, view_dirs

DD = "/media/kavinder/hdd2/liepose_data"; HO = "/media/kavinder/hdd3/larc_figs/holdout"
info = json.load(open(f"{DD}/bop_datasets/tless/models_eval/models_info.json"))
q2m = lambda q: Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()  # noqa: E731
te, tr = np.load(f"{DD}/tless_tokens/test_meta.npz"), np.load(f"{DD}/tless_tokens/train_meta.npz")
Rte, Rtr = q2m(te["rotation_allo"]), q2m(tr["rotation_allo"])
ote, otr = te["object_id"].astype(int), tr["object_id"].astype(int)
kte = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
ktr = np.load(f"{DD}/tless_tokens/train_crop_meta.npy").astype(np.float32)
tte = te["translation_scaled"].astype(np.float32)
ttr = np.load(f"{DD}/tless_tokens/train_ts_true.npy").astype(np.float32)
cont = np.array([("symmetries_continuous" in info[str(o)]) for o in range(1, 31)])


def flips_z(o):
  for s in info[str(o)].get("symmetries_discrete", []):
    if (np.asarray(s, float).reshape(4, 4)[:3, :3] @ np.array([0, 0, 1.0]))[2] < -0.5:
      return True
  return "symmetries_continuous" in info[str(o)] and abs(info[str(o)]["symmetries_continuous"][0]["axis"][2]) < 0.5


FLIP = np.array([flips_z(o) for o in range(1, 31)])


def elevation(R, obj):
  d = view_dirs(R)
  e = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
  return np.where(FLIP[obj - 1], np.abs(e), e)


def roll(R, obj):
  d = view_dirs(R); c = R[:, 1, :]  # camera y-axis in the object frame (perpendicular to d)
  ref = np.where(np.abs(d[:, 2:3]) > 0.95, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
  u = ref - (ref * d).sum(1, keepdims=True) * d; u /= np.linalg.norm(u, axis=1, keepdims=True)
  r = np.degrees(np.arctan2((np.cross(c, u) * d).sum(1), (c * u).sum(1)))
  return np.where(FLIP[obj - 1], np.mod(r, 180.0), np.mod(r, 360.0))


def distance(R, t, k):
  geo = TlessGeometry(); out = []
  f = jax.jit(lambda R_, t_, k_: geo.cam_pose(R_, t_, k_)[1])
  for s in range(0, len(R), 4096):
    out.append(np.asarray(f(jnp.asarray(R[s:s + 4096], jnp.float32), jnp.asarray(t[s:s + 4096]), jnp.asarray(k[s:s + 4096]))))
  return np.linalg.norm(np.concatenate(out), axis=1)


def wrapdiff(a, b, period):
  d = np.mod(a - b, period); return np.minimum(d, period - d)


dirs = np.load(f"{HO}/dirs.npz")["dirs"]
regions = {}
for deg in (60, 90):
  tr_in = in_holdout_cone(otr, Rtr, dirs, deg, info) & ~cont[otr - 1]
  te_in = in_holdout_cone(ote, Rte, dirs, deg, info) & ~cont[ote - 1]
  regions[f"cone{deg}"] = (tr_in, te_in, f"{deg}-degree cones, objects without continuous symmetry")
e_te, e_tr = elevation(Rte, ote), elevation(Rtr, otr)
thr = float(np.percentile(e_te, 75))
regions["elev"] = (e_tr > thr, e_te > thr, f"elevation > {thr:.1f} deg")
d_te, d_tr = distance(Rte, tte, kte), distance(Rtr, ttr, ktr)
thr_d = float(np.percentile(d_te, 70))
regions["dist"] = (d_tr > thr_d, d_te > thr_d, f"distance > {thr_d:.3f} m")
r_te, r_tr = roll(Rte, ote), roll(Rtr, otr)
tr_in, te_in = np.zeros(len(otr), bool), np.zeros(len(ote), bool)
for o in range(1, 31):
  P = 180.0 if FLIP[o - 1] else 360.0
  med = np.median(r_te[ote == o])
  tr_in[otr == o] = wrapdiff(r_tr[otr == o], med, P) < 30
  te_in[ote == o] = wrapdiff(r_te[ote == o], med, P) < 30
regions["roll"] = (tr_in, te_in, "in-plane rotation within 30 deg of the median test roll")
summary = {}
for nm, (tr_in, te_in, desc) in regions.items():
  os.makedirs(f"{HO}/{nm}", exist_ok=True)
  np.save(f"{HO}/{nm}/exclude_rows.npy", np.nonzero(tr_in)[0]); np.save(f"{HO}/{nm}/test_in.npy", te_in)
  kept = [1 - tr_in[otr == o].mean() for o in range(1, 31)]
  summary[nm] = dict(desc=desc, train_removed=int(tr_in.sum()), train_total=len(tr_in), test_in=int(te_in.sum()),
                     min_kept_frac_per_object=round(float(min(kept)), 3))
  print(nm, summary[nm], flush=True)
np.savez(f"{HO}/region_stats.npz", elev_te=e_te, elev_tr=e_tr, dist_te=d_te, dist_tr=d_tr, roll_te=r_te, roll_tr=r_tr)
json.dump(summary, open(f"{HO}/regions.json", "w"), indent=1)
print("test elevation pct 10/50/90", np.percentile(e_te, [10, 50, 90]).round(1), "train", np.percentile(e_tr, [10, 50, 90]).round(1))
print("test distance pct 10/50/90", np.percentile(d_te, [10, 50, 90]).round(3), "train", np.percentile(d_tr, [10, 50, 90]).round(3))
