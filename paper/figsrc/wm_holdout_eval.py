"""Held-out viewpoints: the frozen field refit without any training crop whose view lies in a 30-degree cone per object
(tless_field --holdout), scored on the test crops, against the field fitted on all views (tl_field_v1), in the frozen
DINOv2 space. Distance = symmetry-aware rotation to the nearest *remaining* training view.

  prediction    fraction of variance unexplained at the GT pose on visible-object patches: full field, held-out field,
                this crop's own mean token, the nearest remaining training view appearance-matched to the crop
  energy        P(E(perturbed) > E(GT)) for rotations of the GT by delta about 8 random axes, test crops in the cones

Run: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python wm_holdout_eval.py
"""
import json
import sys
import types

for _m in ("bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.tless_field import in_holdout_cone, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats

DD = "/media/kavinder/hdd2/liepose_data"; OUT = "/media/kavinder/hdd3/larc_figs/data"; HO = "/media/kavinder/hdd3/larc_figs/holdout"
info = json.load(open(f"{DD}/bop_datasets/tless/models_eval/models_info.json"))
q2m = lambda q: Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()  # noqa: E731


def rot_dist(Rq, Rs, o):
  v = info[str(o)]
  if "symmetries_continuous" in v:
    ax = np.asarray(v["symmetries_continuous"][0]["axis"], float)
    e = np.degrees(np.arccos(np.clip(np.einsum("nij,j->ni", Rs, ax) @ (Rq @ ax), -1, 1)))
    return np.minimum(e, 180 - e) if v.get("symmetries_discrete") else e
  Ss = [np.eye(3)] + [np.asarray(s, np.float64).reshape(4, 4)[:3, :3] for s in v.get("symmetries_discrete", [])]
  return np.min(np.stack([np.degrees(np.arccos(np.clip((np.einsum("ij,nij->n", Rq @ S, Rs) - 1) / 2, -1, 1))) for S in Ss]), 0)


tm = np.load(f"{DD}/tless_tokens/test_meta.npz"); N = len(tm["object_id"])
obj = tm["object_id"].astype(int); R = q2m(tm["rotation_allo"]); t = tm["translation_scaled"].astype(np.float32)
k = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
z = np.asarray(np.load(f"{DD}/tless_tokens/test_tokens.npy", mmap_mode="r"), np.float32)
fg = np.load(f"{DD}/tless_tokens/test_fg.npy").astype(np.float32)
trm = np.load(f"{DD}/tless_tokens/train_meta.npz"); Rtr = q2m(trm["rotation_allo"]); otr = trm["object_id"].astype(int)
held = np.zeros(len(otr), bool); held[np.load(f"{HO}/tl_field_ho30/holdout_rows.npy")] = True
dirs = np.load(f"{HO}/dirs.npz")["dirs"]
cone = in_holdout_cone(obj, R, dirs, 30.0, info)

nn, dnn = np.zeros(N, int), np.zeros(N)
for o in range(1, 31):
  rows = np.where((otr == o) & ~held)[0]
  for i in np.where(obj == o)[0]:
    d = rot_dist(R[i], Rtr[rows], o); j = int(np.argmin(d)); nn[i], dnn[i] = rows[j], d[j]
print("nearest remaining view, in-cone test crops: pct 50/90/max", np.percentile(dnn[cone], [50, 90, 100]).round(1), flush=True)

ztr = np.load(f"{DD}/tless_tokens/train_tokens.npy", mmap_mode="r"); u = np.unique(nn)
zn = np.asarray(ztr[u], np.float32)[np.searchsorted(u, nn)]
zn = (zn - zn.mean(1, keepdims=True)) / (zn.std(1, keepdims=True) + 1e-6) * z.std(1, keepdims=True) + z.mean(1, keepdims=True)
field, fp_full = load_tless_field("/media/kavinder/hdd3/larp_abl/probe/tl_field_v1")
_, fp_ho = load_tless_field(f"{HO}/tl_field_ho30")
Rf = R.astype(np.float32)


def render(fp, Rs, idx):
  f = jax.jit(lambda o, R_, t_, k_, zz: field.apply(fp, o, R_, t_, k_, app_stats(zz)))
  return np.concatenate([np.asarray(f(jnp.asarray(obj[idx][s:s + 128]), Rs[s:s + 128], t[idx][s:s + 128], k[idx][s:s + 128], z[idx][s:s + 128]))
                         for s in range(0, len(idx), 128)])


ia = np.arange(N)
mu = (fg[..., None] * z).sum((0, 1)) / fg.sum()
var = (fg * ((z - mu) ** 2).sum(-1)).sum(1) + 1e-8
fvu = lambda P: (fg * ((P - z) ** 2).sum(-1)).sum(1) / var  # noqa: E731
E = {"full_field": fvu(render(fp_full, Rf, ia)), "heldout_field": fvu(render(fp_ho, Rf, ia)),
     "own_mean_token": fvu(np.broadcast_to(z.mean(1, keepdims=True), z.shape)), "nearest_remaining_view_matched": fvu(zn)}
ok = fg.sum(1) > 2
res = {"n_test": int(ok.sum()), "n_in_cone": int((ok & cone).sum()), "by_distance": {}, "in_cone": {}, "outside_cone": {}}
for nm, s in (("in_cone", ok & cone), ("outside_cone", ok & ~cone)):
  res[nm] = {kk: float(np.median(v[s])) for kk, v in E.items()}
  res[nm]["heldout_better_than_matched_view"] = float((E["heldout_field"][s] < E["nearest_remaining_view_matched"][s]).mean())
  res[nm]["heldout_better_than_own_mean"] = float((E["heldout_field"][s] < E["own_mean_token"][s]).mean())
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 181)):
  s = ok & (dnn >= lo) & (dnn < hi)
  if s.sum():
    res["by_distance"][f"{lo}-{hi}"] = {"n": int(s.sum()), **{kk: float(np.median(v[s])) for kk, v in E.items()}}
print(json.dumps(res, indent=1), flush=True)

# energy discrimination inside the cones
rng = np.random.default_rng(3)
ic = np.where(ok & cone)[0]
ic = np.sort(rng.choice(ic, min(600, len(ic)), replace=False))
disc = {}
for nm, fp in (("full_field", fp_full), ("heldout_field", fp_ho)):
  f = jax.jit(lambda o, R_, t_, k_, zz, w: weighted_energy(field.apply(fp, o, R_, t_, k_, app_stats(zz)), zz, w))
  e_gt = np.concatenate([np.asarray(f(jnp.asarray(obj[ic][s:s + 128]), Rf[ic][s:s + 128], t[ic][s:s + 128], k[ic][s:s + 128],
                                      z[ic][s:s + 128], FG_FLOOR + fg[ic][s:s + 128])) for s in range(0, len(ic), 128)])
  disc[nm] = {}
  for dl in (5, 10, 20, 45, 90):
    wins = []
    for rep in range(8):
      ax = rng.normal(size=(len(ic), 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
      Rp = (Rotation.from_rotvec(ax * np.radians(dl)).as_matrix() @ R[ic]).astype(np.float32)
      e_p = np.concatenate([np.asarray(f(jnp.asarray(obj[ic][s:s + 128]), Rp[s:s + 128], t[ic][s:s + 128], k[ic][s:s + 128],
                                         z[ic][s:s + 128], FG_FLOOR + fg[ic][s:s + 128])) for s in range(0, len(ic), 128)])
      sym_ok = np.array([rot_dist(Rp[j].astype(np.float64), R[ic][j][None], obj[ic][j])[0] > 0.5 * dl for j in range(len(ic))])
      wins.append((e_p > e_gt)[sym_ok])
    disc[nm][str(dl)] = float(np.concatenate(wins).mean())
res["energy_P(E_perturbed>E_gt)_in_cone"] = disc
print(json.dumps(disc, indent=1), flush=True)
json.dump(res, open(f"{OUT}/wm_holdout30.json", "w"), indent=1)
print("saved")
