"""Extrapolation tests: a frozen field refit without one held-out region (wm_holdout_regions.py), scored at the GT pose on the
test crops inside the region, against the field fitted on all views, in the frozen DINOv2 space (FVU on visible-object
cells, as wm_holdout_eval.py). Binned by the symmetry-aware rotation to the nearest remaining training view, and for the
elevation / distance regions also by how far the crop lies beyond the region's threshold. Energy discrimination as before.

Run: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python wm_holdout_eval2.py NAME
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
from liepose.exp.jepa_probe.tless_field import load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats

NAME = sys.argv[1]
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
held = np.zeros(len(otr), bool); held[np.load(f"{HO}/{NAME}/exclude_rows.npy")] = True
inside = np.load(f"{HO}/{NAME}/test_in.npy")
ok = fg.sum(1) > 2
idx = np.where(inside & ok)[0]

nn, dnn = np.zeros(len(idx), int), np.zeros(len(idx))
for o in np.unique(obj[idx]):
  rows = np.where((otr == o) & ~held)[0]
  for j in np.where(obj[idx] == o)[0]:
    d = rot_dist(R[idx[j]], Rtr[rows], o); m = int(np.argmin(d)); nn[j], dnn[j] = rows[m], d[m]
ztr = np.load(f"{DD}/tless_tokens/train_tokens.npy", mmap_mode="r"); u = np.unique(nn)
zi = z[idx]
zn = np.asarray(ztr[u], np.float32)[np.searchsorted(u, nn)]
zn = (zn - zn.mean(1, keepdims=True)) / (zn.std(1, keepdims=True) + 1e-6) * zi.std(1, keepdims=True) + zi.mean(1, keepdims=True)
field, fp_full = load_tless_field("/media/kavinder/hdd3/larp_abl/probe/tl_field_v1")
_, fp_ho = load_tless_field(f"{HO}/{NAME}/field")
Rf = R.astype(np.float32)


def render(fp):
  f = jax.jit(lambda o, R_, t_, k_, zz: field.apply(fp, o, R_, t_, k_, app_stats(zz)))
  return np.concatenate([np.asarray(f(jnp.asarray(obj[idx][s:s + 128]), Rf[idx][s:s + 128], t[idx][s:s + 128], k[idx][s:s + 128],
                                      zi[s:s + 128])) for s in range(0, len(idx), 128)])


mu = (fg[..., None] * z).sum((0, 1)) / fg.sum()
w = fg[idx]
var = (w * ((zi - mu) ** 2).sum(-1)).sum(1) + 1e-8
fvu = lambda P: (w * ((P - zi) ** 2).sum(-1)).sum(1) / var  # noqa: E731
E = {"full_field": fvu(render(fp_full)), "heldout_field": fvu(render(fp_ho)),
     "own_mean_token": fvu(np.broadcast_to(zi.mean(1, keepdims=True), zi.shape)), "nearest_remaining_view_matched": fvu(zn)}
res = {"region": NAME, "n_in": int(len(idx)),
       "nearest_remaining_view_deg_pct_50_90_max": np.percentile(dnn, [50, 90, 100]).round(1).tolist(),
       "median": {kk: float(np.median(v)) for kk, v in E.items()},
       "heldout_better_than_matched_view": float((E["heldout_field"] < E["nearest_remaining_view_matched"]).mean()),
       "heldout_better_than_own_mean": float((E["heldout_field"] < E["own_mean_token"]).mean()), "by_distance": {}}
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90), (90, 181)):
  s = (dnn >= lo) & (dnn < hi)
  if s.sum() >= 10:
    res["by_distance"][f"{lo}-{hi}"] = {"n": int(s.sum()), **{kk: float(np.median(v[s])) for kk, v in E.items()}}
st = np.load(f"{HO}/region_stats.npz")
for key, col, edges in (("elev", "elev_te", None), ("dist", "dist_te", None)):
  if NAME == key:
    x = st[col][idx]; q = np.percentile(x, [0, 25, 50, 75, 100]); q[-1] += 1e-6
    res["by_" + key] = {f"{q[i]:.3f}-{q[i + 1]:.3f}": {"n": int(((x >= q[i]) & (x < q[i + 1])).sum()),
                        **{kk: float(np.median(v[(x >= q[i]) & (x < q[i + 1])])) for kk, v in E.items()}} for i in range(4)}
print(json.dumps(res, indent=1), flush=True)

rng = np.random.default_rng(3)
ic = np.sort(rng.choice(len(idx), min(600, len(idx)), replace=False))
gi = idx[ic]
disc = {}
for nm, fp in (("full_field", fp_full), ("heldout_field", fp_ho)):
  f = jax.jit(lambda o, R_, t_, k_, zz, ww: weighted_energy(field.apply(fp, o, R_, t_, k_, app_stats(zz)), zz, ww))
  en = lambda Rs: np.concatenate([np.asarray(f(jnp.asarray(obj[gi][s:s + 128]), Rs[s:s + 128], t[gi][s:s + 128], k[gi][s:s + 128],  # noqa
                                                z[gi][s:s + 128], FG_FLOOR + fg[gi][s:s + 128])) for s in range(0, len(gi), 128)])
  e_gt = en(Rf[gi]); disc[nm] = {}
  for dl in (5, 10, 20, 45, 90):
    wins = []
    for rep in range(8):
      ax = rng.normal(size=(len(gi), 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
      Rp = (Rotation.from_rotvec(ax * np.radians(dl)).as_matrix() @ R[gi]).astype(np.float32)
      sym_ok = np.array([rot_dist(Rp[j].astype(np.float64), R[gi][j][None], obj[gi][j])[0] > 0.5 * dl for j in range(len(gi))])
      wins.append((en(Rp) > e_gt)[sym_ok])
    disc[nm][str(dl)] = float(np.concatenate(wins).mean())
res["energy_P(E_perturbed>E_gt)"] = disc
print(json.dumps(disc, indent=1), flush=True)
json.dump(res, open(f"{OUT}/wm_holdout_{NAME}.json", "w"), indent=1)
print("saved")
