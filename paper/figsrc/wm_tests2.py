"""Further world-model tests (read-only use of trained fields; T-LESS test set scored once; nothing tuned on test).
Stage-1 field (frozen DINOv2 space) unless noted; the jointly trained field in its own space where marked.

  transl     ranking under translation perturbations: shift the GT translation along x, y (in-plane) or z (depth) by a
             fraction of the object diameter (converted to the crop parametrisation), P(E(perturbed) > E(GT)) per size
  corresp    token correspondence across viewpoints (cross-view pairs of wm_world): for each visible cell p of B, is the
             observed token of B most similar (cosine) to the rendered token at the same cell? top-1 and within one cell,
             by rotation between views; controls: A's tokens (no change), the field at A's pose
  appear     prediction at B's GT pose with appearance codes from: B itself, another image of the same scene, an image
             of the same object in another scene, the dataset mean code (no image)
  occl       cross-view error (saved pairs) binned by the change of visible fraction between A and B
  likelihood residuals at the GT pose (excess kurtosis per channel, correlation across cells); the temperature tau of
             p(z|g) ~ exp(-E/tau) fitted on 500 held-out TRAINING crops (pairwise preference likelihood), then the
             calibration of P(a closer than b) = sigmoid((E_b - E_a) / tau) on test pairs (ECE), and the Gaussian tau
Run: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python wm_tests2.py
"""
import glob
import json
import sys
import types

for _m in ("bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.tless_field import TlessGeometry, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats

DD = "/media/kavinder/hdd2/liepose_data"; OUT = "/media/kavinder/hdd3/larc_figs/data"; TL = f"{DD}/bop_datasets/tless"
info = json.load(open(f"{TL}/models_eval/models_info.json"))
q2m = lambda q: Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()  # noqa: E731
tm = np.load(f"{DD}/tless_tokens/test_meta.npz"); N = len(tm["object_id"])
obj = tm["object_id"].astype(int); R = q2m(tm["rotation_allo"]).astype(np.float32); t = tm["translation_scaled"].astype(np.float32)
k = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
z = np.asarray(np.load(f"{DD}/tless_tokens/test_tokens.npy", mmap_mode="r"), np.float32)
fg = np.load(f"{DD}/tless_tokens/test_fg.npy").astype(np.float32)
sig = np.load(f"{OUT}/test_scene_im_gt.npy")
field, fp = load_tless_field("/media/kavinder/hdd3/larp_abl/probe/tl_field_v1")
geo = TlessGeometry()
res = {}


def rend(R_, t_, k_, o_, app, bs=128):
  f = jax.jit(lambda o, R1, t1, k1, a1: field.apply(fp, o, R1, t1, k1, a1))
  return np.concatenate([np.asarray(f(jnp.asarray(o_[s:s + bs]), R_[s:s + bs], t_[s:s + bs], k_[s:s + bs], app[s:s + bs]))
                         for s in range(0, len(o_), bs)])


APP = np.asarray(app_stats(jnp.asarray(z)))  # (N, 768) per-crop appearance statistics
mu = (fg[..., None] * z).sum((0, 1)) / fg.sum()
var = (fg * ((z - mu) ** 2).sum(-1)).sum(1) + 1e-8


def fvu(P, idx):
  return (fg[idx] * ((P - z[idx]) ** 2).sum(-1)).sum(1) / var[idx]


def energy(P, idx):
  return np.asarray(weighted_energy(jnp.asarray(P), jnp.asarray(z[idx]), jnp.asarray(FG_FLOOR + fg[idx])))


ok = fg.sum(1) > 2
rng = np.random.default_rng(11)

# ------------------------------------------------------------------ transl: translation perturbations
idx = np.sort(rng.choice(np.where(ok)[0], 800, replace=False))
_, tc = geo.cam_pose(jnp.asarray(R[idx]), jnp.asarray(t[idx]), jnp.asarray(k[idx]))
tc = np.asarray(tc)
diam = np.array([info[str(o)]["diameter"] / 1000.0 for o in obj[idx]])
e_gt = energy(rend(R[idx], t[idx], k[idx], obj[idx], APP[idx]), idx)
# numerical Jacobian of the crop parametrisation w.r.t. the camera translation, per crop (finite differences)
J = np.zeros((len(idx), 3, 3))
for a in range(3):
  dt = np.zeros(3, np.float32); dt[a] = 1e-3
  _, tp = geo.cam_pose(jnp.asarray(R[idx]), jnp.asarray(t[idx] + dt), jnp.asarray(k[idx]))
  J[:, :, a] = (np.asarray(tp) - tc) / 1e-3
Jinv = np.linalg.inv(J)
out = {}
for axis, nm in ((0, "x"), (1, "y"), (2, "z (depth)")):
  out[nm] = {}
  for frac in (0.02, 0.05, 0.1, 0.2, 0.5):
    wins = []
    for sgn in (1, -1):
      dcam = np.zeros((len(idx), 3)); dcam[:, axis] = sgn * frac * diam
      t_p = (t[idx] + np.einsum("nij,nj->ni", Jinv, dcam)).astype(np.float32)
      wins.append(energy(rend(R[idx], t_p, k[idx], obj[idx], APP[idx]), idx) > e_gt)
    out[nm][str(frac)] = float(np.concatenate(wins).mean())
res["transl_P(E_pert>E_gt)_by_fraction_of_diameter"] = out
print("transl", json.dumps(out), flush=True)

# ------------------------------------------------------------------ cross-view pairs (as wm_world)
cv = np.load(f"{OUT}/wm_world_crossview_frozen_frozen.npz")
A, B, drot = cv["A"], cv["B"], cv["drot"]
sel = np.sort(rng.choice(len(A), 6000, replace=False))
A, B, drot = A[sel], B[sel], drot[sel]
pred_A = rend(R[B], t[B], k[B], obj[B], APP[A])
pred_Apose = rend(R[A], t[B], k[B], obj[B], APP[A])


def corresp(P, Bi):
  """per pair: fraction of visible cells of B whose observed token is most similar to the prediction at the same cell
  (top-1 among B's visible cells) and within one cell"""
  top1, near = [], []
  for j in range(len(Bi)):
    vis = np.where(fg[Bi[j]] > 0.5)[0]
    if len(vis) < 4:
      top1.append(np.nan); near.append(np.nan); continue
    o = z[Bi[j]][vis]; p = P[j][vis]
    o = o / np.linalg.norm(o, axis=1, keepdims=True); p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    am = (p @ o.T).argmax(1)  # for each predicted cell, the most similar observed visible cell
    top1.append((am == np.arange(len(vis))).mean())
    rr, cc = vis // 8, vis % 8
    near.append(((np.abs(rr[am] - rr) <= 1) & (np.abs(cc[am] - cc) <= 1)).mean())
  return np.array(top1), np.array(near)


C = {"field (A's appearance)": corresp(pred_A, B), "copy A": corresp(z[A], B), "field at A's pose": corresp(pred_Apose, B)}
out = {}
for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 90), (90, 181)):
  s = (drot >= lo) & (drot < hi)
  out[f"{lo}-{hi}"] = {"n": int(s.sum()), **{nm: [round(float(np.nanmean(v[0][s])), 3), round(float(np.nanmean(v[1][s])), 3)]
                                              for nm, v in C.items()}}
res["corresp_top1_within1_by_rotation"] = out
print("corresp", json.dumps(out), flush=True)

# ------------------------------------------------------------------ appear: appearance code sources
idx = np.sort(rng.choice(np.where(ok)[0], 1500, replace=False))
same, other = np.full(len(idx), -1), np.full(len(idx), -1)
for j, i in enumerate(idx):
  c1 = np.where((obj == obj[i]) & (sig[:, 0] == sig[i, 0]) & (sig[:, 1] != sig[i, 1]))[0]
  c2 = np.where((obj == obj[i]) & (sig[:, 0] != sig[i, 0]))[0]
  if len(c1):
    same[j] = rng.choice(c1)
  if len(c2):
    other[j] = rng.choice(c2)
trm = np.load(f"{DD}/tless_tokens/train_meta.npz")
ztr = np.load(f"{DD}/tless_tokens/train_tokens.npy", mmap_mode="r")
tr_s = np.sort(rng.choice(len(trm["object_id"]), 20000, replace=False))
mean_app = np.asarray(app_stats(jnp.asarray(np.asarray(ztr[tr_s], np.float32)))).mean(0)
m = (same >= 0) & (other >= 0)
ii = idx[m]
E = {"own image": fvu(rend(R[ii], t[ii], k[ii], obj[ii], APP[ii]), ii),
     "other image, same scene": fvu(rend(R[ii], t[ii], k[ii], obj[ii], APP[same[m]]), ii),
     "image of another scene": fvu(rend(R[ii], t[ii], k[ii], obj[ii], APP[other[m]]), ii),
     "dataset mean (no image)": fvu(rend(R[ii], t[ii], k[ii], obj[ii], np.broadcast_to(mean_app, APP[ii].shape).copy()), ii),
     "mean token (appearance only)": fvu(np.broadcast_to(z[ii].mean(1, keepdims=True), z[ii].shape), ii)}
res["appear_fvu_at_gt"] = {nm: float(np.median(v)) for nm, v in E.items()}
res["appear_n"] = int(len(ii))
print("appear", json.dumps(res["appear_fvu_at_gt"]), flush=True)

# ------------------------------------------------------------------ occl: cross-view error by change of visibility
vf = np.zeros(N)
gi = {}
for f in glob.glob(f"{TL}/test_primesense/*/scene_gt_info.json"):
  sc = int(f.split("/")[-2]); d = json.load(open(f))
  for im, lst in d.items():
    for g, x in enumerate(lst):
      gi[(sc, int(im), g)] = x["visib_fract"]
for i in range(N):
  vf[i] = gi[(int(sig[i, 0]), int(sig[i, 1]), int(sig[i, 2]))]
cvf = np.load(f"{OUT}/wm_world_crossview_frozen_frozen.npz")
dv = np.abs(vf[cvf["B"]] - vf[cvf["A"]])
out = {}
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
  s = (dv >= lo) & (dv < hi)
  out[f"{lo}-{hi}"] = {"n": int(s.sum()), "field": float(np.median(cvf["field_A_appearance"][s])),
                       "copy A": float(np.median(cvf["copy_A"][s])), "field at A's pose": float(np.median(cvf["field_at_A_pose"][s]))}
out["by_visible_fraction_of_B"] = {f"{lo}-{hi}": float(np.median(cvf["field_A_appearance"][(vf[cvf['B']] >= lo) & (vf[cvf['B']] < hi)]))
                                   for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01))}
res["occl_crossview_fvu_by_visibility_change"] = out
print("occl", json.dumps(out), flush=True)

# ------------------------------------------------------------------ likelihood
idx = np.sort(rng.choice(np.where(ok)[0], 1000, replace=False))
P = rend(R[idx], t[idx], k[idx], obj[idx], APP[idx])
r = (P - z[idx])[fg[idx] > 0.5]  # (cells, D) residuals on visible cells
rs = (r - r.mean(0)) / (r.std(0) + 1e-8)
kurt = (rs ** 4).mean(0) - 3
cell_r = np.linalg.norm(P - z[idx], axis=-1)  # (crops, 64)
cc = np.corrcoef(cell_r[:, :-1].ravel(), cell_r[:, 1:].ravel())[0, 1]  # neighbouring cells along the row-major order


def pairs(ids, Rs, ts, ks, os_, zz, ww, app, n_rep=4):
  """energies at pairs of random perturbations (5..90 deg) and their true symmetry-free rotation distances"""
  Ea, Eb, da, db = [], [], [], []
  fe = jax.jit(lambda o, R1, t1, k1, a1, z1, w1: weighted_energy(field.apply(fp, o, R1, t1, k1, a1), z1, w1))
  for rep in range(n_rep):
    ang_a = rng.uniform(2, 90, len(ids)); ang_b = rng.uniform(2, 90, len(ids))
    axa = rng.normal(size=(len(ids), 3)); axa /= np.linalg.norm(axa, axis=1, keepdims=True)
    axb = rng.normal(size=(len(ids), 3)); axb /= np.linalg.norm(axb, axis=1, keepdims=True)
    Ra = (Rotation.from_rotvec(axa * np.radians(ang_a[:, None])).as_matrix() @ Rs).astype(np.float32)
    Rb = (Rotation.from_rotvec(axb * np.radians(ang_b[:, None])).as_matrix() @ Rs).astype(np.float32)
    for Rx, L in ((Ra, Ea), (Rb, Eb)):
      L.append(np.concatenate([np.asarray(fe(jnp.asarray(os_[s:s + 128]), Rx[s:s + 128], ts[s:s + 128], ks[s:s + 128], app[s:s + 128],
                                             jnp.asarray(zz[s:s + 128]), jnp.asarray(ww[s:s + 128]))) for s in range(0, len(ids), 128)]))
    da.append(ang_a); db.append(ang_b)
  return np.concatenate(Ea), np.concatenate(Eb), np.concatenate(da), np.concatenate(db)


# tau on held-out training crops (the field fit's held-out source images; rng 1234 as tless_field)
grp = np.unique(trm["index"]); held = np.random.default_rng(1234).permutation(grp)[: int(0.03 * len(grp))]
hv = np.sort(rng.choice(np.nonzero(np.isin(trm["index"], held))[0], 500, replace=False))
ztv = np.asarray(ztr[hv], np.float32)
fgv = np.load(f"{DD}/tless_tokens/train_fg.npy", mmap_mode="r")[hv].astype(np.float32)
kv = np.load(f"{DD}/tless_tokens/train_crop_meta.npy").astype(np.float32)[hv]
tv = np.load(f"{DD}/tless_tokens/train_ts_true.npy").astype(np.float32)[hv]
Ea, Eb, da, db = pairs(hv, q2m(trm["rotation_allo"][hv]), tv, kv, trm["object_id"][hv], ztv, FG_FLOOR + fgv, np.asarray(app_stats(jnp.asarray(ztv))))
y = (da < db).astype(float)  # a is closer to the truth (symmetric objects: approximate, both measured from the GT)


def nll(lt):
  pa = 1.0 / (1.0 + np.exp(-(Eb - Ea) / np.exp(lt)))
  return -np.mean(y * np.log(pa + 1e-9) + (1 - y) * np.log(1 - pa + 1e-9))


lt = minimize_scalar(nll, bounds=(-12, 2), method="bounded").x
tau = float(np.exp(lt))
Ea, Eb, da, db = pairs(idx, R[idx], t[idx], k[idx], obj[idx], z[idx], FG_FLOOR + fg[idx], APP[idx])
p = 1 / (1 + np.exp(-(Eb - Ea) / tau)); y = (da < db).astype(float)
bins = np.linspace(0, 1, 11); ece = 0.0; rel = []
for lo, hi in zip(bins[:-1], bins[1:]):
  s = (p >= lo) & (p < hi + (1e-9 if hi == 1 else 0))
  if s.sum():
    ece += s.mean() * abs(p[s].mean() - y[s].mean()); rel.append([round(float(p[s].mean()), 3), round(float(y[s].mean()), 3), int(s.sum())])
W_ = (FG_FLOOR + fg[idx]).sum(1).mean(); s2 = float((r ** 2).mean())
res["likelihood"] = dict(excess_kurtosis_median=float(np.median(kurt)), excess_kurtosis_p90=float(np.percentile(kurt, 90)),
                         corr_neighbouring_cell_residuals=float(cc), tau_fitted_on_training=tau,
                         tau_gaussian_iid=float(2 * s2 / (384 * W_)), calibration_ece=float(ece), reliability=rel,
                         pairwise_accuracy=float(((Eb > Ea) == (da < db)).mean()))
print("likelihood", json.dumps(res["likelihood"]), flush=True)
json.dump(res, open(f"{OUT}/wm_tests2.json", "w"), indent=1)
print("saved")
