"""Paper figures from real inference (fig_data.py outputs, BOP per-instance errors, run_select samples). matplotlib only.

Run: JAX_PLATFORMS=cpu BOP_DS=tless DATADIR=/media/kavinder/hdd2/liepose_data PYTHONPATH=LARp/src:LARp/lib/bop_toolkit \
       python figs.py    -> LARp/paper/fig/gen/*.pdf|png
Numbers in the summary figure are copied from the paper's tables (sec/5_experiments.tex), not recomputed.
"""
import glob
import json
import os
import sys
import types

for _m in ("bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
from scipy.spatial.transform import Rotation
from skimage.measure import find_contours

from bop_toolkit_lib import inout
from liepose.exp.jepa_probe.tless_field import TlessGeometry
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR

D = "/media/kavinder/hdd3/larc_figs/data"
ERR = "/media/kavinder/hdd3/larc_figs/errs"
DD = "/media/kavinder/hdd2/liepose_data"
TL = f"{DD}/bop_datasets/tless"
OUT = "/home/kavinder/ARNABI_ARSH/LARp/paper/fig/gen"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
plt.rcParams.update({
  "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
  "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
  "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "axes.spines.top": False,
  "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42, "legend.frameon": False})
C = {"larc": "#1f5fbf", "larc2": "#6aa0e8", "ctrl": "#d1495b", "ctrl2": "#f0a3ad", "frozen": "#8a8a8a",
     "gt": "#ffffff", "sym": "#f2b134", "lie": "#3b9b5a", "ink": "#222222"}
COL1, COL2 = 3.3, 6.9  # CVPR column / page width, inches
RES = LinearSegmentedColormap.from_list("res", ["#0b0b12", "#3b1f6b", "#b5367a", "#f68f46", "#fbe9a6"])


def save(fig, name):
  for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
  plt.close(fig)
  print("saved", name)


def off(ax):
  ax.set_xticks([]); ax.set_yticks([])
  for s in ax.spines.values():
    s.set_visible(False)


info = json.load(open(f"{TL}/models_eval/models_info.json"))
geo = TlessGeometry()
_mesh = {}


def mesh(o):
  o = int(o)
  if o not in _mesh:
    m = inout.load_ply(f"{TL}/models_eval/obj_{o:06d}.ply")
    _mesh[o] = (m["pts"] / 1000.0, np.asarray(m["faces"]).astype(int))
  return _mesh[o]


def silhouette(o, R_allo, t_s, k, res=224, ss=2):
  """projected model silhouette in the crop (display only: CAD is never used by LaRC)"""
  v, f = mesh(o)
  R, t = geo.cam_pose(jnp.asarray(R_allo)[None], jnp.asarray(t_s)[None], jnp.asarray(k)[None])
  Pc = v @ np.asarray(R[0]).T + np.asarray(t[0])
  if (Pc[:, 2] <= 1e-3).any():
    return np.zeros((res, res), bool)
  u = Pc[:, :2] / Pc[:, 2:] * k[6:8] + k[8:10]
  u = (u - k[1:3]) * k[0] * ss
  im = Image.new("L", (res * ss, res * ss), 0)
  dr = ImageDraw.Draw(im)
  for tri in f:
    dr.polygon([tuple(u[i]) for i in tri], fill=255)
  return zoom(np.asarray(im) > 0, 1 / ss, order=0)


def contour(ax, m, color, lw=0.9, ls="-"):
  for c in find_contours(np.pad(m.astype(float), 1), 0.5):
    ax.plot(c[:, 1] - 1, c[:, 0] - 1, color=color, lw=lw, ls=ls)


def pca_basis(X):
  mu = X.mean(0)
  _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
  P = (X - mu) @ Vt[:3].T
  lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
  return lambda a: np.clip(((a - mu) @ Vt[:3].T - lo) / (hi - lo), 0, 1).reshape(8, 8, 3)


def tokimg(ax, rgb):
  ax.imshow(rgb, interpolation="nearest")
  off(ax)


traj = [dict(np.load(f"{D}/traj_{i}.npz")) for i in range(6)]
OBJS = [int(t["obj"]) for t in traj]


def residual(zh, z, w):
  return (w * ((zh.astype(np.float32) - z) ** 2).mean(-1)).reshape(8, 8)


# ================================================================== Fig: render-and-compare at a right and a wrong pose
def fig_render_compare():
  t = traj[0]
  z, w = t["z_joint"], t["w"]
  # a wrong hypothesis visited by the sampler (chain 0): rotation 30-60 deg off with the translation close to the truth
  rr, tt = t["joint_rerr"][:100, 0], t["joint_terr"][:100, 0]
  cand = np.where((rr > 30) & (rr < 60))[0]
  i_wr = int(cand[np.argmin(tt[cand])]) if len(cand) else int(np.argmin(np.abs(rr - 45)))
  print("wrong pose: step", i_wr, "rot", rr[i_wr], "trans err", tt[i_wr])
  zh_wr, zh_gt = t["joint_zh0"][i_wr].astype(np.float32), t["gt_zh_joint"]
  col = pca_basis(np.concatenate([z, zh_gt, zh_wr]))
  r_gt, r_wr = residual(zh_gt, z, w), residual(zh_wr, z, w)
  vmax = np.percentile(np.concatenate([r_gt.ravel(), r_wr.ravel()]), 99)
  E = lambda zh: float((w * ((zh - z) ** 2).mean(-1)).sum() / w.sum())
  s_gt = silhouette(t["obj"], t["R_gt"], t["t_gt"], t["k"])
  s_wr = silhouette(t["obj"], t["joint_R"][i_wr, 0], t["joint_T"][i_wr, 0], t["k"])
  fig, ax = plt.subplots(2, 5, figsize=(COL2, 3.0), gridspec_kw=dict(wspace=0.08, hspace=0.32))
  ax[0, 0].imshow(t["img"]); ax[0, 0].set_title("observed crop $I$"); off(ax[0, 0])
  tokimg(ax[0, 1], col(z)); ax[0, 1].set_title(r"observed tokens $\mathbf{z}$")
  ax[1, 0].imshow(t["img"] * t["mask"][..., None]); contour(ax[1, 0], s_gt, C["gt"], ls="--"); contour(ax[1, 0], s_wr, C["sym"])
  ax[1, 0].set_title("$g^\\star$ (dashed), wrong $g$"); off(ax[1, 0])
  ax[1, 1].imshow(w.reshape(8, 8), cmap="gray", vmin=0, vmax=1.05); off(ax[1, 1]); ax[1, 1].set_title("token weights $w_p$")
  for j, (nm, zh, r, op, c) in enumerate((("$g^\\star$", zh_gt, r_gt, t["gt_op_joint"], C["gt"]),
                                          (f"wrong $g$ ({rr[i_wr]:.0f}$^\\circ$ off)", zh_wr, r_wr, t["joint_op0"][i_wr], C["sym"]))):
    a = ax[j, 2]; tokimg(a, col(zh)); a.set_title(r"render $\hat{\mathbf{z}}(g)$ at " + nm)
    a = ax[j, 3]; a.imshow(op.reshape(8, 8), cmap="gray", vmin=0, vmax=1); off(a); a.set_title(r"opacity $\mathbf{o}(g)$")
    a = ax[j, 4]; im = a.imshow(r, cmap=RES, vmin=0, vmax=vmax); off(a)
    a.set_title(rf"residual, $E$={E(zh):.3f}")
  for j, lab in enumerate(("(a)", "(b)")):
    ax[j, 0].text(-0.08, 1.0, lab, transform=ax[j, 0].transAxes, fontweight="bold", va="top", ha="right")
  save(fig, "fig_render_compare")


# ================================================================== Fig: the comparison inside the loop, step by step
def fig_in_loop(ci=4):
  t = traj[ci]
  z, w = t["z_joint"], t["w"]
  steps = [0, 50, 75, 88, 95, 99]
  ch = 0
  col = pca_basis(np.concatenate([z, t["gt_zh_joint"]] + [t["joint_zh0"][s].astype(np.float32) for s in steps]))
  res_all = [residual(t["joint_zh0"][s], z, w) for s in steps]
  vmax = np.percentile(np.concatenate([r.ravel() for r in res_all]), 99)
  sig = np.load(f"{D}/traj_{ci}.npz")  # noqa (same file)
  fig = plt.figure(figsize=(COL2, 3.6))
  gs = fig.add_gridspec(4, len(steps) + 1, height_ratios=[1, 1, 1, 1.25], hspace=0.16, wspace=0.05)
  s_gt = silhouette(t["obj"], t["R_gt"], t["t_gt"], t["k"])
  for j, s in enumerate(steps):
    a = fig.add_subplot(gs[0, j]); a.imshow((t["img"] * (0.35 + 0.65 * t["mask"][..., None])).astype(np.uint8))
    contour(a, s_gt, C["gt"], 0.7, "--"); contour(a, silhouette(t["obj"], t["joint_R"][s, ch], t["joint_T"][s, ch], t["k"]), C["sym"], 1.0)
    a.set_xlim(0, 223); a.set_ylim(223, 0); off(a)
    a.set_title(f"step {s + 1}/100\n{t['joint_rerr'][s, ch]:.0f}$^\\circ$ off", fontsize=6.5)
    a = fig.add_subplot(gs[1, j]); tokimg(a, col(t["joint_zh0"][s].astype(np.float32)))
    a = fig.add_subplot(gs[2, j]); a.imshow(res_all[j], cmap=RES, vmin=0, vmax=vmax); off(a)
  a = fig.add_subplot(gs[1, len(steps)]); tokimg(a, col(z)); a.set_title(r"observed $\mathbf{z}$", fontsize=6.5)
  fig.text(0.075, 0.80, "hypothesis $g_t$", rotation=90, va="center", ha="center")
  fig.text(0.075, 0.62, r"render $\hat{\mathbf{z}}(g_t)$", rotation=90, va="center", ha="center")
  fig.text(0.075, 0.45, r"residual", rotation=90, va="center", ha="center")
  a = fig.add_subplot(gs[3, :3])
  x = np.arange(101)
  for c in range(t["joint_rerr"].shape[1]):
    a.plot(x, t["joint_rerr"][:, c], color=C["larc"], alpha=0.35, lw=0.7)
  a.plot(x, t["joint_rerr"][:, ch], color=C["sym"], lw=1.3, label="chain shown above")
  a.set_yscale("log"); a.set_ylim(0.3, 200); a.set_xlabel("denoising step"); a.set_ylabel("rotation error (deg)")
  a.legend(loc="lower left")
  a = fig.add_subplot(gs[3, 4:])
  xe = np.arange(0, 101, 5)
  Eg = float((w * ((t["gt_zh_joint"] - z) ** 2).mean(-1)).sum() / w.sum())
  for c in range(t["joint_E"].shape[1]):
    a.plot(xe, t["joint_E"][:, c] / Eg, color=C["larc"], alpha=0.35, lw=0.7)
  a.axhline(1, color=C["ink"], lw=0.6, ls="--"); a.text(55, 1.15, "energy at $g^\\star$", fontsize=6)
  a.set_xlabel("denoising step"); a.set_ylabel(r"$E(g_t)\,/\,E(g^\star)$")
  save(fig, "fig_in_loop")


# ================================================================== Fig: trajectories, LaRC vs control, six objects
def fig_trajectories():
  fig, ax = plt.subplots(2, 6, figsize=(COL2, 2.5), gridspec_kw=dict(hspace=0.12, wspace=0.12, height_ratios=[1, 1.2]))
  for j, t in enumerate(traj):
    a = ax[0, j]; a.imshow(t["img"]); off(a); a.set_title(f"object {t['obj']}")
    a = ax[1, j]
    x = np.arange(101)
    for nm, c in (("ctrl", C["ctrl"]), ("joint", C["larc"])):
      for k in range(t[f"{nm}_rerr"].shape[1]):
        a.plot(x, t[f"{nm}_rerr"][:, k], color=c, alpha=0.3, lw=0.6)
      a.plot(x, np.median(t[f"{nm}_rerr"], 1), color=c, lw=1.4, label={"ctrl": "control", "joint": "LaRC"}[nm])
    a.set_yscale("log"); a.set_ylim(0.3, 200); a.set_xticks([0, 50, 100])
    if j == 0:
      a.set_ylabel("rotation error (deg)"); a.legend(loc="lower left")
    else:
      a.set_yticklabels([])
    a.set_xlabel("step")
  save(fig, "fig_trajectories")


# ================================================================== Fig: energy landscape and discrimination
def fig_energy():
  L = np.load(f"{D}/landscape.npz")
  th, ph = np.degrees(L["theta"]), np.degrees(L["phi"])
  dsc = np.load(f"{D}/disc.npz")
  fig = plt.figure(figsize=(COL2, 2.0))
  gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.25, 1, 1], wspace=0.85)
  for j, (nm, title) in enumerate((("frozen", "frozen DINOv2 field"), ("joint", "LaRC (joint) field"))):
    a = fig.add_subplot(gs[0, j])
    E = L[nm] / L[nm][len(ph) // 2, len(th) // 2]
    im = a.imshow(E, extent=[th[0], th[-1], ph[0], ph[-1]], origin="lower", aspect="auto", cmap="viridis",
                  vmin=np.percentile(E, 1), vmax=np.percentile(E, 99))
    a.plot(0, 0, marker="+", color="w", ms=8, mew=1.4); a.plot([-180, 180], [0, 0], "o", mfc="none", mec=C["sym"], ms=6, mew=1.2)
    a.set_title(title, loc="left"); a.set_xlabel(r"rotation about sym. axis (deg)")
    if j == 0:
      a.set_ylabel("tilt (deg)")
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=5.5)
    a.text(1.0, 1.02, r"$E/E(g^\star)$", transform=a.transAxes, fontsize=6, ha="left")
  dl = dsc["deltas"]
  a = fig.add_subplot(gs[0, 2])
  b = fig.add_subplot(gs[0, 3])
  for nm, c, lab in (("frozen", C["frozen"], "frozen"), ("joint", C["larc"], "LaRC (joint)")):
    X = dsc[nm]; E, re = X[:, 0], X[:, 1]
    r = E / E[:, :1, :1]
    med = np.median(r, axis=(0, 2)); q1, q3 = np.percentile(r, 25, axis=(0, 2)), np.percentile(r, 75, axis=(0, 2))
    a.plot(dl[1:], med[1:], "-o", color=c, ms=2.5, label=lab); a.fill_between(dl[1:], q1[1:], q3[1:], color=c, alpha=0.18, lw=0)
    win = (E > E[:, :1, :1])
    bins = [(1, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 90), (90, 181)]
    fr = [win[(re >= lo) & (re < hi)].mean() for lo, hi in bins]
    b.plot(np.arange(len(bins)), np.array(fr) * 100, "-o", color=c, ms=2.5, label=lab)
  a.set_xscale("log"); a.set_xlabel(r"perturbation $\delta$ (deg)"); a.set_ylabel(r"median $E/E(g^\star)$")
  a.legend(loc="upper left"); a.set_title("energy rises away from $g^\\star$")
  b.set_xticks(range(7)); b.set_xticklabels(["1", "2", "5", "10", "20", "45", "90"], fontsize=5.5)
  b.set_xlabel("distance to $g^\\star$ (deg)"); b.set_ylabel(r"% with $E > E(g^\star)$"); b.set_ylim(40, 101)
  b.set_title("ground truth ranked first")
  save(fig, "fig_energy")


# ================================================================== Fig: emergent geometry (density never supervised)
def fig_geometry():
  d = np.load(f"{D}/density.npz")
  n = len(d["grid"])
  fig, ax = plt.subplots(4, 6, figsize=(COL2, 4.35), gridspec_kw=dict(hspace=0.06, wspace=0.04))
  for j, t in enumerate(traj):
    o = int(t["obj"])
    ax[0, j].imshow(t["img"]); off(ax[0, j]); ax[0, j].set_title(f"object {o}")
    sig = d[f"joint_{o}"]
    dz = 2.0 * float(d[f"r_{o}"]) / n
    for i, axis in enumerate((2, 1, 0)):
      sg = np.moveaxis(sig, axis, -1)[..., ::-1]  # rays along -axis (viewer on the + side)
      cum = np.cumsum(sg * dz, -1)
      hit = cum >= 1.0  # first sample where the ray is 63% opaque
      first = np.where(hit.any(-1), hit.argmax(-1), n)
      depth = np.where(first < n, first / n, np.nan)
      a = ax[i + 1, j]
      a.imshow(np.ma.masked_invalid(1 - depth).T, origin="lower", cmap="magma", vmin=0.2, vmax=1.0); off(a)
      a.set_facecolor("black")
  for i, lab in enumerate(("test crop", "field depth, $z$", "field depth, $y$", "field depth, $x$")):
    ax[i, 0].text(-0.1, 0.5, lab, rotation=90, transform=ax[i, 0].transAxes, va="center", ha="right")
  save(fig, "fig_geometry")


# ================================================================== per-instance BOP errors
DIAM = {int(k): v["diameter"] for k, v in info.items()}


def load_err(name, err):
  out = {}
  for f in glob.glob(f"{ERR}/{name}/error={err}_ntop=-1/errors_*.json"):
    sc = int(os.path.basename(f)[7:13])
    for e in json.load(open(f)):
      ev = min(min(v) for v in e["errors"].values())
      out[(sc, e["im_id"], e["obj_id"], e["est_id"])] = ev
  return out


RUNS = [("joint_consensus_n16", "LaRC + consensus (16)", C["larc"], "-"),
        ("joint_n1", "LaRC", C["larc"], "--"),
        ("field_full_lat_ft", "LaRC w/o joint training", C["larc2"], "-"),
        ("frozen_field_full_lat", "LaRC, frozen DINOv2", C["frozen"], "-"),
        ("field_nowm_ft", "control (trained encoder)", C["ctrl"], "-"),
        ("frozen_field_nowm", "control, frozen DINOv2", C["ctrl2"], "-")]


def fig_accuracy():
  fig, ax = plt.subplots(1, 2, figsize=(COL2 * 0.62, 1.75), gridspec_kw=dict(wspace=0.32))
  for name, lab, c, ls in RUNS:
    m = load_err(name, "mssd")
    keys = list(m)
    e = np.array([m[k] / DIAM[k[2]] for k in keys])
    th = np.linspace(0, 0.5, 101)
    ax[0].plot(th, [(e < x).mean() * 100 for x in th], color=c, ls=ls, label=lab)
    p = load_err(name, "mspd")
    e2 = np.array(list(p.values()))
    th2 = np.linspace(0, 50, 101)
    ax[1].plot(th2, [(e2 < x).mean() * 100 for x in th2], color=c, ls=ls)
    print(name, "n", len(e), "MSSD<0.1d", round((e < 0.1).mean() * 100, 1), "MSPD<10px", round((e2 < 10).mean() * 100, 1))
  for a, lo, hi in ((ax[0], 0.05, 0.5), (ax[1], 5, 50)):
    a.axvspan(lo, hi, color="#000000", alpha=0.04, lw=0)
  ax[0].set_xlabel("MSSD threshold (fraction of diameter)"); ax[0].set_ylabel("recall (%)")
  ax[1].set_xlabel("MSPD threshold (px)")
  for a in ax:
    a.set_ylim(0, 100)
  ax[1].legend(*ax[0].get_legend_handles_labels(), loc="lower right", fontsize=5.4, handlelength=1.6)
  save(fig, "fig_accuracy")


def fig_per_object():
  th_s = np.arange(0.05, 0.51, 0.05)
  th_p = np.arange(5, 51, 5)
  res = {}
  for name in ("joint_n1", "field_nowm_ft"):
    ms, mp = load_err(name, "mssd"), load_err(name, "mspd")
    per = {}
    for k in ms:
      o = k[2]
      per.setdefault(o, []).append(((ms[k] / DIAM[o] < th_s).mean() + (mp.get(k, 1e9) < th_p).mean()) / 2)
    res[name] = {o: np.mean(v) * 100 for o, v in per.items()}
  objs = sorted(res["joint_n1"])
  x = np.arange(len(objs))
  fig, a = plt.subplots(figsize=(COL2, 1.45))
  a.bar(x - 0.2, [res["field_nowm_ft"][o] for o in objs], 0.4, color=C["ctrl"], label="control (trained encoder)")
  a.bar(x + 0.2, [res["joint_n1"][o] for o in objs], 0.4, color=C["larc"], label="LaRC")
  a.set_xticks(x); a.set_xticklabels(objs, fontsize=5.8); a.set_xlabel("T-LESS object")
  a.set_ylabel(r"(AR$_\mathrm{MSSD}$+AR$_\mathrm{MSPD}$)/2"); a.set_ylim(0, 100); a.set_xlim(-0.7, len(objs) - 0.3)
  a.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
  save(fig, "fig_per_object")


# ================================================================== multi-hypothesis and selection (full test set)
def sym_mats(o):
  v = info[str(o)]
  S = [np.eye(3)] + [np.asarray(s, np.float64).reshape(4, 4)[:3, :3] for s in v.get("symmetries_discrete", [])]
  return np.stack(S), (np.asarray(v["symmetries_continuous"][0]["axis"], float) if "symmetries_continuous" in v else None)


def rot_err(Rgt, Rm, o):
  S, ax = sym_mats(o)
  if ax is not None:  # continuous symmetry: angle between the symmetry axes
    a1 = Rgt @ ax; a2 = np.einsum("...ij,j->...i", Rm, ax)
    e = np.degrees(np.arccos(np.clip(a2 @ a1, -1, 1)))
    if len(S) > 1:
      e = np.minimum(e, 180 - e)
    return e
  e = []
  for Sm in S:
    M = np.einsum("ij,...jk->...ik", (Rgt @ Sm).T, Rm)
    e.append(np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1))))
  return np.min(np.stack(e), 0)


def wxyz(q):
  return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()


def fig_selection():
  s = np.load(f"{D}/a1_sel16_samples.npz")
  meta = np.load(f"{DD}/tless_tokens/test_meta.npz")
  row_of = {int(i): r for r, i in enumerate(meta["index"])}
  rows = np.array([row_of[int(i)] for i in s["index"]])
  obj = meta["object_id"][rows]
  Rg = wxyz(meta["rotation_allo"][rows])
  R = wxyz(s["pose"][..., :4].reshape(-1, 4)).reshape(len(rows), 16, 3, 3)
  E = s["energy"]
  err = np.stack([rot_err(Rg[i], R[i], int(obj[i])) for i in range(len(rows))])  # (N, 16)
  ok = err < 10
  # consensus: most other samples within 10 deg (symmetry-aware, as run_select)
  cons = np.zeros(len(rows), int)
  for i in range(len(rows)):
    Sm, ax = sym_mats(int(obj[i]))
    A = np.stack([rot_err(R[i, a], R[i], int(obj[i])) for a in range(16)])
    cons[i] = int(np.argmax((A < 10).sum(1)))
  ie = E.argmin(1)
  pick = {"first sample": ok[:, 0], "consensus": ok[np.arange(len(rows)), cons], "lowest energy": ok[np.arange(len(rows)), ie],
          "oracle": ok.any(1)}
  print({k: round(v.mean() * 100, 1) for k, v in pick.items()})
  fig, ax = plt.subplots(1, 3, figsize=(COL2, 1.9), gridspec_kw=dict(wspace=0.42, width_ratios=[1.1, 1, 1]))
  # (a) how many of the 16 samples are correct, per crop
  a = ax[0]
  nk = ok.sum(1)
  h = np.bincount(nk, minlength=17) / len(nk) * 100
  a.bar(np.arange(17), h, color=[C["ctrl"]] + [C["larc2"]] * 15 + [C["larc"]], width=0.85)
  a.set_yscale("log"); a.set_ylim(0.1, 100)
  a.set_xlabel(r"samples within $10^\circ$ of $g^\star$ (of 16)"); a.set_ylabel("% of crops")
  a.set_xticks([0, 4, 8, 12, 16]); a.set_title(f"{h[16]:.0f}% all right, {h[0]:.1f}% none")
  # (b) accuracy (rot < 10 deg) of the rules
  a = ax[1]
  names = list(pick)
  vals = [pick[k].mean() * 100 for k in names]
  bars = a.bar(range(4), vals, color=[C["frozen"], C["larc2"], C["larc"], "#cfcfcf"], edgecolor=[C["ink"]] * 3 + ["#999999"], lw=0.4)
  for b_, v in zip(bars, vals):
    a.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=6)
  a.set_xticks(range(4)); a.set_xticklabels(["first", "consensus", "energy", "oracle"], fontsize=6)
  a.set_ylabel(r"% crops with rotation error $<10^\circ$"); a.set_ylim(min(vals) - 10, 100)
  a.set_title("choosing one of 16")
  # (c) confidence: selected sample's energy ranks crops
  a = ax[2]
  sel_ok = pick["lowest energy"]
  agree = np.array([(rot_err(R[i, ie[i]], R[i], int(obj[i])) < 10).sum() for i in range(len(rows))])
  from sklearn.metrics import roc_auc_score
  for conf, lab, c in ((agree, "agreement of the 16", C["larc"]), (-E[np.arange(len(rows)), ie], "energy of the pick", C["frozen"])):
    auc = roc_auc_score(sel_ok, conf)
    o_ = np.argsort(-conf, kind="stable")
    cov = np.arange(1, len(o_) + 1) / len(o_)
    acc = np.cumsum(sel_ok[o_]) / np.arange(1, len(o_) + 1)
    a.plot(cov[50:] * 100, acc[50:] * 100, color=c, label=f"{lab} (AUROC {auc:.2f})")
  a.axhline(sel_ok.mean() * 100, color=C["ink"], lw=0.6, ls="--")
  a.set_xlabel("crops kept (%)"); a.set_ylabel("accuracy of kept (%)"); a.legend(loc="lower left", fontsize=5.6); a.set_ylim(80, 100.5)
  a.set_title("knowing when it is right")
  save(fig, "fig_selection")


# ================================================================== results summary (numbers from the paper's tables)
def fig_summary():
  fig, ax = plt.subplots(1, 3, figsize=(COL2, 1.85), gridspec_kw=dict(wspace=0.45, width_ratios=[1.6, 1, 1]))
  a = ax[0]
  rows = [("LiePose (pub.)", 70.1, None, C["lie"]), ("LiePose (ours, 8x data)", 73.7, None, C["lie"]),
          ("control, frozen", 40.5, 48.6, C["ctrl2"]), ("LaRC, frozen", 72.4, 78.5, C["frozen"]),
          ("control, trained enc.", 52.2, 69.6, C["ctrl"]), ("LaRC w/o joint", 75.3, 85.4, C["larc2"]),
          ("LaRC", 82.2, None, C["larc"]), ("LaRC + consensus", 84.9, None, C["larc"])]
  y = np.arange(len(rows))[::-1]
  for yy, (lab, v, _, c) in zip(y, rows):
    a.barh(yy, v, color=c, height=0.7)
    a.text(v + 1, yy, f"{v:.1f}", va="center", fontsize=6)
  a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=6); a.set_xlim(0, 100); a.set_xlabel("T-LESS AR (GT crops; 100k$\\times$32 unless noted)")
  a = ax[1]
  N = [1, 4, 16]
  for lab, v, c, ls in (("consensus", [82.3, 84.3, 84.9], C["larc2"], "-"), ("energy", [82.3, 84.8, 84.6], C["larc"], "-"),
                        ("oracle", [82.3, 87.3, 89.0], "#9a9a9a", "--")):
    a.plot(N, v, ls, color=c, marker="o", ms=3, label=lab)
  a.set_xscale("log", base=2); a.set_xticks(N); a.set_xticklabels(N); a.set_xlabel("samples $N$"); a.set_ylabel("T-LESS AR")
  a.legend(loc="upper left"); a.set_title("selection (LaRC)")
  a = ax[2]
  st = [10, 20, 100]
  a.plot(st, [58.2, 58.5, 57.8], "-o", color=C["larc"], ms=3, label="LaRC")
  a.plot(st, [34.1, 33.5, 32.7], "-o", color=C["ctrl"], ms=3, label="control")
  a.plot([100], [72.4], "*", color=C["larc"], ms=6); a.plot([100], [40.5], "*", color=C["ctrl"], ms=6)
  a.set_xscale("log"); a.set_xticks(st); a.set_xticklabels(st); a.set_xlabel("sampler steps"); a.set_ylabel("T-LESS AR")
  a.set_ylim(20, 80); a.legend(loc="center right"); a.set_title("frozen enc.: flow matching ($\\bullet$)\nvs diffusion ($\\star$)", fontsize=6.5)
  save(fig, "fig_summary")


# ================================================================== second set (fig_data2.py, errors, samples)
def fig_views():
  """the world model imagines unseen views: rendered latent (PCA) and opacity on a turntable, one PCA basis per object"""
  V = np.load(f"{D}/views.npz")
  az = np.degrees(V["az"]).astype(int)
  show = [0, 2, 4]  # three of the six objects (fixed order of fig_data)
  fig = plt.figure(figsize=(COL2, 3.15))
  gs = fig.add_gridspec(2 * len(show), len(az) + 1, hspace=0.05, wspace=0.04, width_ratios=[1.15] + [1] * len(az))
  for i, ci in enumerate(show):
    t = traj[ci]
    zh = V[f"zh_{ci}"].astype(np.float32)[len(az):2 * len(az)]  # elevation 0
    op = V[f"op_{ci}"][len(az):2 * len(az)]
    col = pca_basis(np.concatenate([t["z_joint"]] + list(zh)))
    a = fig.add_subplot(gs[2 * i:2 * i + 2, 0]); a.imshow(t["img"]); off(a)
    a.set_title(f"object {t['obj']}" if i else f"observed (object {t['obj']})", fontsize=6.5)
    for j in range(len(az)):
      a = fig.add_subplot(gs[2 * i, j + 1]); tokimg(a, col(zh[j]))
      if i == 0:
        a.set_title(f"{az[j]}$^\\circ$", fontsize=6.5)
      a = fig.add_subplot(gs[2 * i + 1, j + 1]); a.imshow(op[j].reshape(8, 8), cmap="gray", vmin=0, vmax=1); off(a)
  fig.text(0.53, 0.995, r"field render $\hat{\mathbf{z}}$ (top) and opacity $\mathbf{o}$ (bottom), object turned about the vertical axis",
           ha="center", va="bottom", fontsize=6.8)
  save(fig, "fig_views")


def fig_occlusion():
  """occlusion: the render is amodal, the residual marks the occluder, the WM weights recover the visible part"""
  O = np.load(f"{D}/occlusion.npz")
  n = len(O["rows"])
  cols = ["crop", "annotation mask", r"opacity $\mathbf{o}(g^\star)$", "residual", "WM weights", "WM wts, no mask"]
  fig, ax = plt.subplots(n, len(cols), figsize=(COL2, 1.2 * n), gridspec_kw=dict(hspace=0.06, wspace=0.04))
  up = lambda x: zoom(x.reshape(8, 8), 28, order=0)
  for j in range(n):
    img, m, sil = O[f"img_{j}"], O[f"mask_{j}"], O[f"sil_{j}"]
    r = O[f"masked_res_{j}"]
    a = ax[j, 0]; a.imshow(img); contour(a, sil, C["sym"], 0.8, "--"); off(a)
    a.text(4, 16, f"obj {int(O[f'obj_{j}'])}, {100 * O['vf'][j]:.0f}% visible", color="w", fontsize=5.5)
    a = ax[j, 1]; a.imshow(O[f"w_{j}"].reshape(8, 8), cmap="gray", vmin=0, vmax=1.05); off(a)
    a = ax[j, 2]; a.imshow(O[f"masked_op_{j}"].reshape(8, 8), cmap="gray", vmin=0, vmax=1); off(a)
    a = ax[j, 3]; a.imshow(r.reshape(8, 8), cmap=RES, vmin=0, vmax=np.percentile(r, 99)); off(a)
    a = ax[j, 4]; a.imshow(O[f"masked_wm_{j}"].reshape(8, 8), cmap="gray", vmin=0, vmax=1.05); off(a)
    a = ax[j, 5]; a.imshow(O[f"raw_wm_{j}"].reshape(8, 8), cmap="gray", vmin=0, vmax=1.05); off(a)
  for i, c in enumerate(cols):
    ax[0, i].set_title(c, fontsize=6.3)
  save(fig, "fig_occlusion")


def fig_population():
  """48 seeded crops x 8 chains: rotation and translation error and LaRC energy along the sampler, LaRC vs control"""
  P = np.load(f"{D}/population.npz")
  x = np.arange(1, 101)
  fig, ax = plt.subplots(1, 4, figsize=(COL2, 1.7), gridspec_kw=dict(wspace=0.75))
  for nm, c, lab in (("ctrl", C["ctrl"], "control"), ("joint", C["larc"], "LaRC")):
    for a, key in ((ax[0], "rerr"), (ax[2], "terr")):
      v = P[f"{nm}_{key}"].transpose(1, 0, 2).reshape(100, -1)  # (step, crop*chain)
      a.plot(x, np.median(v, 1), color=c, label=lab)
      a.fill_between(x, np.percentile(v, 25, 1), np.percentile(v, 75, 1), color=c, alpha=0.18, lw=0)
    re = P[f"{nm}_rerr"].transpose(1, 0, 2).reshape(100, -1)
    ax[1].plot(x, (re < 10).mean(1) * 100, color=c, label=lab)
    E = P[f"{nm}_E"].transpose(1, 0, 2).reshape(100, -1)
    ax[3].plot(x, np.median(E, 1), color=c, label=lab)
    print(nm, "final: rot<10", (re[-1] < 10).mean().round(3), "median terr", np.median(P[f"{nm}_terr"][:, -1]).round(4))
  ax[0].set_yscale("log"); ax[0].set_ylabel("rotation error (deg)")
  ax[1].set_ylabel(r"% chains $<10^\circ$"); ax[1].set_ylim(0, 100)
  ax[2].set_yscale("log"); ax[2].set_ylabel("translation error (scaled)")
  ax[3].set_ylabel(r"LaRC energy $E(g_t)$"); ax[3].set_yscale("log")
  for a in ax:
    a.set_xlabel("denoising step"); a.set_xticks([1, 50, 100])
  ax[1].legend(loc="upper left")
  save(fig, "fig_population")


def fig_gallery(n=8):
  """the first n population crops (seed order, no selection): GT (dashed), LaRC and control, chain 0"""
  P = np.load(f"{D}/population.npz")
  meta = np.load(f"{DD}/tless_tokens/test_meta.npz")
  sh = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
  fig, ax = plt.subplots(2, n // 2 * 2 // 2 if False else n, figsize=(COL2, 2.0), gridspec_kw=dict(hspace=0.05, wspace=0.04))
  from matplotlib.lines import Line2D
  for j in range(n):
    r, o, k = int(P["rows"][j]), int(P["obj"][j]), P["k"][j]
    s_gt = silhouette(o, P["R_gt"][j], P["t_gt"][j], k)
    for i, (nm, c) in enumerate((("joint", C["larc"]), ("ctrl", C["ctrl"]))):
      a = ax[i, j]; a.imshow(sh["image"][r]); off(a); a.set_xlim(0, 223); a.set_ylim(223, 0)
      contour(a, s_gt, C["gt"], 0.7, "--")
      contour(a, silhouette(o, P[f"{nm}_R"][j, 0], P[f"{nm}_T"][j, 0], k), c, 1.0)
      e = P[f"{nm}_rerr"][j, -1, 0]
      a.text(6, 214, f"{e:.0f}$^\\circ$", color="w", fontsize=6, fontweight="bold")
  ax[0, 0].text(-0.08, 0.5, "LaRC", rotation=90, transform=ax[0, 0].transAxes, va="center", ha="right")
  ax[1, 0].text(-0.08, 0.5, "control", rotation=90, transform=ax[1, 0].transAxes, va="center", ha="right")
  fig.legend([Line2D([], [], color="k", ls="--", lw=0.8), Line2D([], [], color=C["larc"]), Line2D([], [], color=C["ctrl"])],
             ["ground truth", "LaRC", "control"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
  save(fig, "fig_gallery")


def _gt_info():
  out = {}
  for f in glob.glob(f"{TL}/test_primesense/*/scene_gt_info.json"):
    sc = int(f.split("/")[-2])
    gi, gt = json.load(open(f)), json.load(open(f.replace("_info", "")))
    for im, lst in gi.items():
      for g, (a, b) in enumerate(zip(lst, gt[im])):
        out[(sc, int(im), g)] = (a["visib_fract"], b["cam_t_m2c"][2])
  return out


def _inst_scores(name):
  """per GT instance: (AR_MSSD + AR_MSPD) / 2 at the BOP19 thresholds, keyed (scene, im, gt_id)"""
  th_s, th_p = np.arange(0.05, 0.51, 0.05), np.arange(5, 51, 5)
  res = {}
  for err, th, norm in (("mssd", th_s, True), ("mspd", th_p, False)):
    for f in glob.glob(f"{ERR}/{name}/error={err}_ntop=-1/errors_*.json"):
      sc = int(os.path.basename(f)[7:13])
      for e in json.load(open(f)):
        g, v = min(((int(g), min(v)) for g, v in e["errors"].items()), key=lambda x: x[1])
        v = v / DIAM[e["obj_id"]] if norm else v
        res.setdefault((sc, e["im_id"], g), []).append((v < th).mean())
  return {k: np.mean(v) for k, v in res.items() if len(v) == 2}


def fig_robustness():
  """accuracy vs visible fraction and vs distance, all GT instances of the T-LESS test set (GT crops)"""
  gi = _gt_info()
  runs = [("joint_n1", "LaRC", C["larc"]), ("field_full_lat_ft", "LaRC w/o joint", C["larc2"]),
          ("frozen_field_full_lat", "LaRC, frozen", C["frozen"]), ("field_nowm_ft", "control", C["ctrl"])]
  vb = np.array([0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01])
  fig, ax = plt.subplots(1, 2, figsize=(COL2 * 0.75, 1.7), gridspec_kw=dict(wspace=0.35))
  dq = None
  for name, lab, c in runs:
    s = _inst_scores(name)
    keys = [k for k in s if k in gi]
    v = np.array([gi[k][0] for k in keys]); z = np.array([gi[k][1] for k in keys]); y = np.array([s[k] for k in keys]) * 100
    if dq is None:
      dq = np.percentile(z, [0, 20, 40, 60, 80, 100]); dq[-1] += 1
      cnt = np.histogram(v, vb)[0]
    xv = [(y[(v >= lo) & (v < hi)].mean()) for lo, hi in zip(vb[:-1], vb[1:])]
    ax[0].plot(np.arange(len(xv)), xv, "-o", color=c, ms=2.5, label=lab)
    xd = [(y[(z >= lo) & (z < hi)].mean()) for lo, hi in zip(dq[:-1], dq[1:])]
    ax[1].plot(np.arange(len(xd)), xd, "-o", color=c, ms=2.5, label=lab)
    print(name, "n", len(y), "by visib", np.round(xv, 1))
  ax[0].set_xticks(range(len(vb) - 1)); ax[0].set_xticklabels([f"{a:.0%}".replace("%", "") for a in vb[:-1]], fontsize=5.5)
  ax[0].set_xlabel("visible fraction (bin start, %)"); ax[0].set_ylabel("instance AR (%)"); ax[0].set_ylim(0, 100)
  for i, c_ in enumerate(cnt):
    ax[0].text(i, 2, str(c_), ha="center", fontsize=4.8, color="#777777")
  ax[1].set_xticks(range(5)); ax[1].set_xticklabels([f"{dq[i] / 1000:.2f}" for i in range(5)], fontsize=5.5)
  ax[1].set_xlabel("distance quintile (start, m)"); ax[1].set_ylim(0, 100)
  ax[1].legend(loc="lower left", fontsize=5.4)
  save(fig, "fig_robustness")


def _samples():
  s = np.load(f"{D}/a1_sel16_samples.npz")
  meta = np.load(f"{DD}/tless_tokens/test_meta.npz")
  row_of = {int(i): r for r, i in enumerate(meta["index"])}
  rows = np.array([row_of[int(i)] for i in s["index"]])
  R = wxyz(s["pose"][..., :4].reshape(-1, 4)).reshape(len(rows), 16, 3, 3)
  return rows, meta["object_id"][rows], wxyz(meta["rotation_allo"][rows]), R, s


def fig_spread():
  """the 16 samples: spread across symmetric modes (raw angle) but on the true symmetry orbit (symmetry-aware)"""
  rows, obj, Rg, R, _ = _samples()
  cls = {}
  for o in np.unique(obj):
    v = info[str(o)]
    cls[int(o)] = "continuous" if "symmetries_continuous" in v else ("discrete" if v.get("symmetries_discrete") else "none")
  raw, sym, cl = [], [], []
  for i in range(len(rows)):
    M = np.einsum("aji,bjk->abik", R[i], R[i])
    ang = np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    raw.append(np.median(ang[np.triu_indices(16, 1)]))
    sym.append(np.median(rot_err(Rg[i], R[i], int(obj[i]))))
    cl.append(cls[int(obj[i])])
  raw, sym, cl = np.array(raw), np.array(sym), np.array(cl)
  fig, ax = plt.subplots(1, 2, figsize=(COL2 * 0.75, 1.6), gridspec_kw=dict(wspace=0.35))
  bins = np.logspace(-1, np.log10(180), 40)
  for k, c in (("none", C["larc"]), ("discrete", C["sym"]), ("continuous", C["ctrl"])):
    m = cl == k
    ax[0].hist(raw[m], bins, histtype="step", color=c, density=True, label=f"{k} ({m.sum()})")
    ax[1].hist(sym[m], bins, histtype="step", color=c, density=True)
    print(k, m.sum(), "median raw spread", np.median(raw[m]).round(1), "median sym err", np.median(sym[m]).round(2))
  for a, t in ((ax[0], "pairwise angle between samples"), (ax[1], "symmetry-aware error to $g^\\star$")):
    a.set_xscale("log"); a.set_xlabel(t + " (deg)"); a.set_yticks([])
  ax[0].set_ylabel("density (median per crop)"); ax[0].legend(title="object symmetry", fontsize=5.4, title_fontsize=5.6, loc="upper left")
  save(fig, "fig_spread")


def fig_failures(n=6):
  """where LaRC (consensus of 16) is wrong: error angles, whether a good sample existed, and examples"""
  rows, obj, Rg, R, s = _samples()
  meta = np.load(f"{DD}/tless_tokens/test_meta.npz")
  sh = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
  kmeta = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
  err = np.stack([rot_err(Rg[i], R[i], int(obj[i])) for i in range(len(rows))])
  cons = np.array([int(np.argmax((np.stack([rot_err(R[i, a], R[i], int(obj[i])) for a in range(16)]) < 10).sum(1)))
                   for i in range(len(rows))])
  e_c = err[np.arange(len(rows)), cons]
  fail = e_c >= 15
  any_ok = (err < 10).any(1)
  print("failures", fail.sum(), "of", len(fail), "with a good sample among 16:", (fail & any_ok).sum())
  fig = plt.figure(figsize=(COL2, 3.0))
  gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.06)
  a = fig.add_subplot(gs[0, :n // 2 + 1])
  a.hist(e_c[fail], np.arange(15, 181, 5), color=C["larc"])
  a.set_xlabel("rotation error of failures (deg)"); a.set_ylabel("crops"); a.set_xticks([15, 45, 90, 135, 180])
  a.set_title(f"{fail.sum()} failures ({100 * fail.mean():.1f}% of crops)", fontsize=6.8)
  a = fig.add_subplot(gs[0, n // 2 + 2:])
  cats = {"one of 16\nright": (fail & any_ok).sum(), "all 16\nwrong": (fail & ~any_ok).sum()}
  a.bar(range(2), list(cats.values()), color=[C["larc2"], C["ctrl"]], width=0.6)
  a.set_xticks(range(2)); a.set_xticklabels(list(cats), fontsize=6); a.set_ylabel("failed crops")
  a.set_title("could selection have fixed it?", fontsize=6.8)
  fi = np.where(fail)[0][:n]  # first failures in test order, no selection
  for j, i in enumerate(fi):
    r, o, k = int(rows[i]), int(obj[i]), kmeta[rows[i]]
    a = fig.add_subplot(gs[1, j]); a.imshow(sh["image"][r]); off(a); a.set_xlim(0, 223); a.set_ylim(223, 0)
    contour(a, silhouette(o, Rg[i], meta["translation_scaled"][r], k), C["gt"], 0.7, "--")
    contour(a, silhouette(o, R[i, cons[i]], s["pose"][i, cons[i], 4:], k), C["larc"], 1.0)
    a.set_title(f"obj {o}: {e_c[i]:.0f}$^\\circ$", fontsize=6.3)
  save(fig, "fig_failures")


def fig_mask_bias():
  """detector masks vs annotation masks: shrunken masks push predicted depth back (diagnostic for Sec. 4)"""
  import csv
  B = f"{TL}/test_primesense"
  cache = {}

  def ratios(path):
    out = []
    for r in csv.DictReader(open(path)):
      s_, i_, o_ = int(r["scene_id"]), int(r["im_id"]), int(r["obj_id"])
      if s_ not in cache:
        cache[s_] = (json.load(open(f"{B}/{s_:06d}/scene_gt.json")), json.load(open(f"{B}/{s_:06d}/scene_gt_info.json")))
      t = np.array([float(x) for x in r["t"].split()])
      cand = [(np.array(g["cam_t_m2c"]), gi["visib_fract"]) for g, gi in zip(cache[s_][0][str(i_)], cache[s_][1][str(i_)]) if g["obj_id"] == o_]
      g, v = min(cand, key=lambda x: np.linalg.norm(x[0][:2] / x[0][2] - t[:2] / t[2]))
      out.append((t[2] / g[2], v))
    return np.array(out)

  gtc = ratios("/media/kavinder/hdd3/larp_abl/experiments/tless-field/train_steps=100000+seed=42+den_mode=field_full+field=tl_field_v1/test_100000/inferences_tless-test.csv")
  det = ratios("/media/kavinder/hdd3/larp_defdet/experiments_vis/tless-field/train_steps=100000+seed=42+den_mode=field_full_lat_ft+field=tl_field_det+joint=0.999/test_100000_sel16/n1/inferences_tless-test.csv")
  fig, ax = plt.subplots(1, 2, figsize=(COL2 * 0.75, 1.6), gridspec_kw=dict(wspace=0.35))
  bins = np.linspace(0.85, 1.25, 61)
  ax[0].hist(gtc[:, 0], bins, histtype="step", color=C["frozen"], density=True, label="annotation masks")
  ax[0].hist(det[:, 0], bins, histtype="step", color=C["ctrl"], density=True, label="detector masks")
  ax[0].axvline(1, color=C["ink"], lw=0.5, ls="--"); ax[0].set_yticks([])
  ax[0].set_xlabel("predicted / true depth"); ax[0].legend(loc="upper right", fontsize=5.4)
  vb = [0, 0.5, 0.8, 0.95, 1.01]
  for X, c, lab in ((gtc, C["frozen"], "annotation"), (det, C["ctrl"], "detector")):
    ax[1].plot(range(4), [np.median(X[(X[:, 1] >= a) & (X[:, 1] < b), 0]) for a, b in zip(vb[:-1], vb[1:])], "-o", color=c, ms=2.5)
  ax[1].axhline(1, color=C["ink"], lw=0.5, ls="--"); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["0-50", "50-80", "80-95", "95+"], fontsize=5.5)
  ax[1].set_xlabel("visible fraction (%)"); ax[1].set_ylabel("median depth ratio")
  save(fig, "fig_mask_bias")


# ================================================================== Fig 1: teaser (page 1, full width)
def fig_teaser_strip(ci=1, ch=None, steps=(0, 60, 99)):
  """left: pose-only diffusion (control) final chains; middle: LaRC's render-and-compare at three steps of one chain;
  right: LaRC final chains; far right: population translation error along the sampler (fig_population data)"""
  t = traj[ci]; o = int(t["obj"]); z, w = t["z_joint"], t["w"]
  if ch is None:
    ch = int(np.argmin(np.abs(t["joint_rerr"][-1] - np.median(t["joint_rerr"][-1]))))  # the median chain, not the best
  zh = [t["joint_zh0"][s].astype(np.float32) for s in steps]
  col = pca_basis(np.concatenate([z, t["gt_zh_joint"]] + zh))
  res = [residual(x, z, w) for x in zh]
  vmax = np.percentile(np.concatenate([r.ravel() for r in res]), 99)
  dim = (t["img"] * (0.35 + 0.65 * t["mask"][..., None])).astype(np.uint8)
  s_gt = silhouette(o, t["R_gt"], t["t_gt"], t["k"])
  fig = plt.figure(figsize=(COL2, 2.05))
  gs = fig.add_gridspec(3, 9, width_ratios=[2.4, 0.18, 1, 1, 1, 0.22, 2.4, 0.95, 2.3], height_ratios=[1, 1, 1],
                        wspace=0.06, hspace=0.08)

  def final(cell, pfx, c, title):
    a = fig.add_subplot(cell); a.imshow(dim)
    for k in range(t[f"{pfx}_R"].shape[1]):
      contour(a, silhouette(o, t[f"{pfx}_R"][-1, k], t[f"{pfx}_T"][-1, k], t["k"]), c, 0.45)
    contour(a, s_gt, C["gt"], 0.8, "--")
    a.set_xlim(0, 223); a.set_ylim(223, 0); off(a); a.set_title(title, fontsize=7)
    a.text(4, 218, f"median: {np.median(t[f'{pfx}_rerr'][-1]):.1f}$^\\circ$, "
                   f"$\\Delta t$ {np.median(t[f'{pfx}_terr'][-1]):.2f}", color="w", fontsize=5.6, va="bottom")
    return a
  final(gs[:, 0], "ctrl", C["ctrl"], "(a) pose-only diffusion\n16 samples")
  for j, s in enumerate(steps):
    a = fig.add_subplot(gs[0, 2 + j]); a.imshow(dim)
    contour(a, s_gt, C["gt"], 0.5, "--")
    contour(a, silhouette(o, t["joint_R"][s, ch], t["joint_T"][s, ch], t["k"]), C["sym"], 0.9)
    a.set_xlim(0, 223); a.set_ylim(223, 0); off(a)
    a.set_title(f"step {s + 1}: {t['joint_rerr'][s, ch]:.0f}$^\\circ$", fontsize=6)
    a = fig.add_subplot(gs[1, 2 + j]); tokimg(a, col(zh[j]))
    a = fig.add_subplot(gs[2, 2 + j]); a.imshow(res[j], cmap=RES, vmin=0, vmax=vmax); off(a)
  final(gs[:, 6], "joint", C["larc"], "(c) LaRC\n16 samples")
  a = fig.add_subplot(gs[:, 8])
  P = np.load(f"{D}/population.npz"); x = np.arange(1, 101)
  for nm, c, lab in (("ctrl", C["ctrl"], "pose-only"), ("joint", C["larc"], "LaRC")):
    v = P[f"{nm}_terr"].transpose(1, 0, 2).reshape(100, -1)
    a.plot(x, np.median(v, 1), color=c, label=lab)
    a.fill_between(x, np.percentile(v, 25, 1), np.percentile(v, 75, 1), color=c, alpha=0.18, lw=0)
  a.set_yscale("log"); a.set_xlabel("denoising step"); a.set_xticks([1, 50, 100])
  a.set_ylabel("translation error"); a.legend(loc="lower left", fontsize=6)
  a.set_title("(d) 48 crops $\\times$ 8 chains", fontsize=7)
  fig.canvas.draw()
  # row labels and the middle-block title, placed from the axes positions
  ax_mid = [x for x in fig.axes if x.get_subplotspec().colspan.start == 2]
  b0, b1, b2 = (ax_mid[i].get_position() for i in range(3))
  xr = [x for x in fig.axes if x.get_subplotspec().colspan.start == 4][0].get_position().x1
  fig.text((b0.x0 + xr) / 2, b0.y1 + 0.13, "(b) LaRC: world model renders $g_t$, denoiser compares",
           ha="center", fontsize=7)
  for bb, lab in zip((b0, b1, b2), ["hypothesis", "WM render", "residual"]):
    fig.text(bb.x0 - 0.008, (bb.y0 + bb.y1) / 2, lab, rotation=90, ha="right", va="center", fontsize=5.8)
  save(fig, "fig_teaser_strip")


# ================================================================== Fig 1: page-1 poster (architecture, in-loop, qualitative, results)
def fig_teaser(ci=1, steps=(0, 60, 99), qual=(0, 1, 2, 3)):
  from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
  from matplotlib.lines import Line2D
  t = traj[ci]; o = int(t["obj"]); z, w = t["z_joint"], t["w"]
  ch = int(np.argmin(np.abs(t["joint_rerr"][-1] - np.median(t["joint_rerr"][-1]))))
  zh = [t["joint_zh0"][s].astype(np.float32) for s in steps]
  col = pca_basis(np.concatenate([z, t["gt_zh_joint"]] + zh))
  res = [residual(x, z, w) for x in zh]
  vmax = np.percentile(np.concatenate([r.ravel() for r in res]), 99)
  dim = (t["img"] * (0.35 + 0.65 * t["mask"][..., None])).astype(np.uint8)
  s_gt = silhouette(o, t["R_gt"], t["t_gt"], t["k"])
  WT, H, FR = COL2 / 0.80, 4.0, 0.80  # left block (a)-(f) at full column width, plus a right column (g)-(i); scaled to \textwidth in LaTeX
  W = WT * FR
  root = plt.figure(figsize=(WT, H))
  fig, f3 = root.subfigures(1, 2, width_ratios=[FR, 1 - FR], wspace=0)

  def crop_ax(rect, img, outlines=(), title=None, fs=6):
    a = fig.add_axes(rect); a.imshow(img); off(a); a.set_xlim(0, 223); a.set_ylim(223, 0)
    for m, c, lw, ls in outlines:
      contour(a, m, c, lw, ls)
    if title:
      a.set_title(title, fontsize=fs, pad=1.5)
    return a

  def sq(x, y, h):  # square axes rect of height h (figure fraction) at (x, y)
    return [x, y, h * H / W, h]

  def panel(x, y, s):
    fig.text(x, y, s, fontsize=7.5, fontweight="bold", va="top")

  # ---------------------------------------------------------------- (a) dense overview: one denoising step, real tensors
  r_ = H / W
  A = fig.add_axes([0, 0, 1, 1], zorder=-1); A.set_xlim(0, 1); A.set_ylim(0, 1); A.axis("off")

  def box(x0_, y0_, x1_, y1_, fc, ec, txt=None, ty=None, fs=6.0):
    A.add_patch(FancyBboxPatch((x0_, y0_), x1_ - x0_, y1_ - y0_, boxstyle="round,pad=0.003,rounding_size=0.008",
                               fc=fc, ec=ec, lw=0.8))
    if txt:
      A.text((x0_ + x1_) / 2, (y0_ + y1_) / 2 if ty is None else ty, txt, ha="center", va="center", fontsize=fs)

  def arrow(p, q, c=C["ink"], cs="arc3", lw=0.7):
    A.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=5, lw=lw, color=c, connectionstyle=cs,
                                shrinkA=0, shrinkB=0))

  def thumb(x, y, h, img, cmap=None, vmin=None, vmax=None, outlines=(), lab=None, fs=5.4):
    a = fig.add_axes([x, y, h * r_, h]); a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"); off(a)
    if img.shape[0] == 224:
      a.set_xlim(0, 223); a.set_ylim(223, 0)
    for m, c, lw, ls in outlines:
      contour(a, m, c, lw, ls)
    if lab:
      A.text(x + h * r_ / 2, y - 0.006, lab, ha="center", va="top", fontsize=fs)
    return a

  s1 = steps[1]
  sil = silhouette(o, t["joint_R"][s1, ch], t["joint_T"][s1, ch], t["k"])
  PL = np.load(f"{D}/planes.npz")
  op = t["joint_op0"][s1].reshape(8, 8)
  r2 = ((zh[1] - z) ** 2).mean(-1); k2 = np.median(r2)
  wm_w = (FG_FLOOR + np.clip(t["joint_op0"][s1], 0, 1) * k2 / (k2 + r2)).reshape(8, 8)
  T_ = 0.135; tw = T_ * r_
  yt, yb = 0.785, 0.525
  # observation branch
  thumb(0.005, yt, T_, dim, lab="observation crop $I$")
  box(0.098, yt + 0.02, 0.19, yt + T_ - 0.02, "#eef2fa", C["larc"], "image encoder\n$\\psi$ (DINOv2-S)\n$\\bar\\psi=\\mathrm{EMA}(\\psi)$", fs=5.4)
  arrow((0.005 + tw + 0.002, yt + T_ / 2), (0.098, yt + T_ / 2))
  thumb(0.205, yt, T_, col(z), lab="observed $\\bar{\\mathbf{z}}=\\bar\\psi(I)$")
  arrow((0.19, yt + T_ / 2), (0.205, yt + T_ / 2))
  # world-model branch
  thumb(0.005, yb, T_, dim, outlines=[(sil, C["sym"], 0.9, "-")], lab="noisy pose $g_t$")
  box(0.098, yb - 0.035, 0.30, yb + T_ + 0.02, "#fdf3df", C["sym"])
  A.text(0.172, yb + T_ + 0.005, "world model $F$: feature field", ha="center", va="center", fontsize=5.8, fontweight="bold")
  hp = 0.062
  for i, kk in enumerate(("xy", "xz", "yz")):
    thumb(0.107 + i * (hp * r_ + 0.006), yb + 0.05, hp, PL[f"{kk}_{o}"], lab=kk, fs=4.8)
  A.text(0.163, yb + 0.03, "", fontsize=5)
  A.text(0.163, yb - 0.015, "per-object tri-planes", ha="center", va="center", fontsize=5.0)
  box(0.225, yb + 0.035, 0.293, yb + 0.115, "#ffffff", "#bbbbbb", "volume\nrender at $g_t$\n+ app. code", fs=4.9)
  arrow((0.107 + 3 * (hp * r_ + 0.006) - 0.003, yb + 0.05 + hp / 2), (0.225, yb + 0.075))
  arrow((0.005 + tw + 0.002, yb + T_ / 2), (0.098, yb + T_ / 2))
  arrow((0.205 + tw / 2, yt - 0.025), (0.259, yb + 0.115), cs="arc3,rad=0.0")
  A.text(0.262, yt - 0.06, "appearance\nstats of $\\bar{\\mathbf{z}}$", ha="left", va="center", fontsize=4.9, color="#555555")
  thumb(0.315, yb, T_, col(zh[1]), lab="render $\\hat{\\mathbf{z}}(g_t)$")
  thumb(0.315 + tw + 0.008, yb, T_, op, cmap="gray", vmin=0, vmax=1, lab="opacity $o(g_t)$")
  arrow((0.30, yb + T_ / 2), (0.315, yb + T_ / 2))
  # comparison
  xc0, xc1, yc0, yc1 = 0.515, 0.618, 0.69, 0.80
  box(xc0, yc0, xc1, yc1, "#f4f4f4", "#777777",
      "compare tokens\n$[\\bar{\\mathbf{z}},\\ \\hat{\\mathbf{z}},\\ \\hat{\\mathbf{z}}\\!-\\!\\bar{\\mathbf{z}},\\ w,\\ \\mathbf{h}]$", fs=5.4)
  arrow((0.205 + tw + 0.002, yt + T_ / 2), (xc0, yc1 - 0.025), cs="arc3,rad=-0.1")
  arrow((0.315 + 2 * tw + 0.01, yb + T_ * 0.75), (xc0, yc0 + 0.02), cs="arc3,rad=0.15")
  T2 = 0.085
  thumb(0.395, 0.825, T2, res[1], cmap=RES, vmin=0, vmax=vmax, lab="residual $r^2$", fs=5.0)
  thumb(0.395 + T2 * r_ + 0.022, 0.825, T2, wm_w, cmap="gray", vmin=0, vmax=1.05, lab="weights $w$", fs=5.0)
  A.text(0.395, 0.95, "WM token weights  $w=o\\cdot\\kappa^2/(\\kappa^2+r^2)$", fontsize=5.0, va="center", ha="left", color="#555555")
  arrow((0.395 + 2 * T2 * r_ + 0.024, 0.825 + T2 / 2), (xc1 - 0.03, yc1), cs="arc3,rad=-0.3")
  # denoiser and loop
  xd0, xd1, yd0, yd1 = 0.515, 0.618, 0.545, 0.645
  box(xd0, yd0, xd1, yd1, "#eef2fa", C["larc"], "SE(3) denoiser $\\epsilon_\\theta$\n(transformer, $t$)\n$g_t\\to g_{t-1}$", fs=5.4)
  arrow(((xc0 + xc1) / 2, yc0), ((xc0 + xc1) / 2, yd1))
  A.add_patch(FancyArrowPatch(((xd0 + xd1) / 2, yd0), (0.005 + tw / 2, yb - 0.035), arrowstyle="-|>", mutation_scale=6,
                              lw=0.9, color=C["larc"], connectionstyle="angle,angleA=-90,angleB=0,rad=3"))
  A.text(0.40, yb - 0.072, "repeat for $t=T,\\ldots,1$ (16 chains)  $\\rightarrow$  select by consensus / energy",
         ha="center", va="center", fontsize=5.6, color=C["larc"])
  A.text(0.40, yb - 0.098, "training: denoising loss + field fit $E(g^\\star;\\,\\mathrm{sg}[\\bar{\\mathbf{z}}])$, encoder and field jointly",
         ha="center", va="center", fontsize=5.3, color="#555555")
  panel(0.0, 0.995, "(a) LaRC: latent render-and-compare inside every SE(3) diffusion step")

  # ---------------------------------------------------------------- (b) one chain through the sampler
  x0, h = 0.675, 0.12
  panel(x0 - 0.025, 0.995, "(b) sampling one test crop")
  ys = [0.815, 0.685, 0.555]
  for j, s in enumerate(steps):
    xj = x0 + j * (h * H / W + 0.006)
    so = silhouette(o, t["joint_R"][s, ch], t["joint_T"][s, ch], t["k"])
    crop_ax(sq(xj, ys[0], h), dim, [(s_gt, C["gt"], 0.5, "--"), (so, C["sym"], 0.9, "-")],
            f"step {s + 1}: {t['joint_rerr'][s, ch]:.0f}$^\\circ$")
    a = fig.add_axes(sq(xj, ys[1], h)); tokimg(a, col(zh[j]))
    a = fig.add_axes(sq(xj, ys[2], h)); a.imshow(res[j], cmap=RES, vmin=0, vmax=vmax); off(a)
  for y_, lab in zip(ys, ["hypothesis", "WM render", "residual"]):
    fig.text(x0 - 0.006, y_ + h / 2, lab, rotation=90, ha="right", va="center", fontsize=5.6)
  xe = x0 + 3 * (h * H / W + 0.006) + 0.01
  crop_ax(sq(xe, ys[0] - 0.06, 0.18), dim,
          [(silhouette(o, t["joint_R"][-1, k], t["joint_T"][-1, k], t["k"]), C["larc"], 0.45, "-") for k in range(16)]
          + [(s_gt, C["gt"], 0.7, "--")], "LaRC, 16 samples")
  crop_ax(sq(xe, ys[2] - 0.005, 0.18), dim,
          [(silhouette(o, t["ctrl_R"][-1, k], t["ctrl_T"][-1, k], t["k"]), C["ctrl"], 0.45, "-") for k in range(16)]
          + [(s_gt, C["gt"], 0.7, "--")])
  fig.text(xe + 0.18 * H / W / 2, ys[2] - 0.012, "pose-only, 16 samples", ha="center", va="top", fontsize=6)

  # ---------------------------------------------------------------- (c) qualitative, seeded test crops (no selection)
  P = np.load(f"{D}/population.npz")
  sh = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
  panel(0.0, 0.405, "(c) T-LESS test crops")
  hq = 0.145
  for j, q in enumerate(qual):
    r, oq, k = int(P["rows"][q]), int(P["obj"][q]), P["k"][q]
    sg = silhouette(oq, P["R_gt"][q], P["t_gt"][q], k)
    for i, (nm, c) in enumerate((("joint", C["larc"]), ("ctrl", C["ctrl"]))):
      a = crop_ax(sq(0.022 + j * (hq * H / W + 0.004), 0.205 - i * (hq + 0.008), hq), sh["image"][r],
                  [(sg, C["gt"], 0.5, "--"), (silhouette(oq, P[f"{nm}_R"][q, 0], P[f"{nm}_T"][q, 0], k), c, 0.8, "-")])
      a.text(5, 216, f"{P[f'{nm}_rerr'][q, -1, 0]:.0f}$^\\circ$, {P[f'{nm}_terr'][q, -1, 0]:.2f}", color="w", fontsize=4.6)
  fig.text(0.017, 0.205 + hq / 2, "LaRC", rotation=90, ha="right", va="center", fontsize=6, color=C["larc"])
  fig.text(0.017, 0.205 - hq / 2 - 0.008, "pose-only", rotation=90, ha="right", va="center", fontsize=6, color=C["ctrl"])

  # ---------------------------------------------------------------- (d) translation error along the sampler (population)
  a = fig.add_axes([0.43, 0.09, 0.165, 0.27])
  x = np.arange(1, 101)
  for nm, c in (("ctrl", C["ctrl"]), ("joint", C["larc"])):
    v = P[f"{nm}_terr"].transpose(1, 0, 2).reshape(100, -1)
    a.plot(x, np.median(v, 1), color=c)
    a.fill_between(x, np.percentile(v, 25, 1), np.percentile(v, 75, 1), color=c, alpha=0.18, lw=0)
  a.set_yscale("log"); a.set_xticks([1, 50, 100]); a.set_xlabel("denoising step", labelpad=1)
  a.set_yticks([0.1, 0.3, 1, 3]); a.set_yticklabels(["0.1", "0.3", "1", "3"]); a.minorticks_off()
  a.set_ylabel("translation error", labelpad=1)
  panel(0.375, 0.405, "(d) 48 crops$\\times$8 chains")

  # ---------------------------------------------------------------- (e) T-LESS AR (paper tables, GT crops)
  a = fig.add_axes([0.715, 0.09, 0.10, 0.27])
  rows = [("LiePose", 70.1, C["lie"]), ("pose-only", 52.2, C["ctrl"]), ("LaRC", 82.2, C["larc"]), ("+select", 84.9, C["larc2"])]
  for i, (lab, v, c) in enumerate(rows):
    a.bar(i, v, color=c, width=0.75); a.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=5.2)
  a.set_xticks(range(len(rows))); a.set_xticklabels([r[0] for r in rows], rotation=40, ha="right", fontsize=5.4)
  a.set_ylim(40, 92); a.set_ylabel("T-LESS AR", labelpad=1)
  panel(0.645, 0.405, "(e) accuracy")

  # ---------------------------------------------------------------- (f) instance AR by visible fraction
  gi = _gt_info()
  vb = np.array([0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01])
  a = fig.add_axes([0.865, 0.09, 0.135, 0.27])
  for name, c in (("field_nowm_ft", C["ctrl"]), ("joint_n1", C["larc"])):
    sc = _inst_scores(name); keys = [k for k in sc if k in gi]
    v = np.array([gi[k][0] for k in keys]); y = np.array([sc[k] for k in keys]) * 100
    a.plot(range(len(vb) - 1), [y[(v >= lo) & (v < hi)].mean() for lo, hi in zip(vb[:-1], vb[1:])], "-o", color=c, ms=2)
  a.set_xticks(range(len(vb) - 1)); a.set_xticklabels([f"{int(b * 100)}" for b in vb[:-1]], fontsize=5)
  a.set_xlabel("visible (%)", labelpad=1); a.set_ylabel("instance AR", labelpad=1); a.set_ylim(0, 100)
  panel(0.83, 0.405, "(f) occlusion")
  LEG = ([Line2D([], [], color="k", ls="--", lw=0.8), Line2D([], [], color=C["larc"]),
              Line2D([], [], color=C["ctrl"]), Line2D([], [], color=C["sym"])],
             ["ground truth", "LaRC", "pose-only diffusion (same encoder)", "current hypothesis"],
             )
  # ================================================================ right column: more results
  def p3(y_, s_):
    f3.text(0.10, y_, s_, fontsize=7.5, fontweight="bold", va="top", zorder=10)
  # (g) rotation error along the sampler: both models converge
  a = f3.add_axes([0.32, 0.775, 0.64, 0.165])
  for nm, c in (("ctrl", C["ctrl"]), ("joint", C["larc"])):
    v = P[f"{nm}_rerr"].transpose(1, 0, 2).reshape(100, -1)
    a.plot(x, np.median(v, 1), color=c)
    a.fill_between(x, np.percentile(v, 25, 1), np.percentile(v, 75, 1), color=c, alpha=0.18, lw=0)
  a.set_yscale("log"); a.set_xticks([1, 50, 100]); a.set_xlabel("denoising step", labelpad=1)
  a.set_yticks([1, 10, 100]); a.set_yticklabels(["1", "10", "100"]); a.minorticks_off()
  a.set_ylabel("rotation err. ($^\\circ$)", labelpad=1)
  p3(0.995, "(g) rotation error")
  # (h) per-object accuracy, pose-only (x) vs LaRC (y)
  th_s, th_p = np.arange(0.05, 0.51, 0.05), np.arange(5, 51, 5)
  per = {}
  for name in ("field_nowm_ft", "joint_n1"):
    ms, mp = load_err(name, "mssd"), load_err(name, "mspd"); acc = {}
    for kk in ms:
      acc.setdefault(kk[2], []).append(((ms[kk] / DIAM[kk[2]] < th_s).mean() + (mp.get(kk, 1e9) < th_p).mean()) / 2)
    per[name] = {ob: np.mean(v) * 100 for ob, v in acc.items()}
  objs = sorted(per["joint_n1"])
  px = np.array([per["field_nowm_ft"][ob] for ob in objs]); py = np.array([per["joint_n1"][ob] for ob in objs])
  better = int((py > px).sum())
  a = f3.add_axes([0.36, 0.385, 0.25 * H / (WT * (1 - FR)), 0.25])
  a.plot([30, 100], [30, 100], color="#999999", lw=0.6, ls="--")
  a.scatter(px, py, s=6, color=C["larc"], zorder=3, lw=0)
  a.set_xlim(30, 100); a.set_ylim(30, 100); a.set_xticks([40, 70, 100]); a.set_yticks([40, 70, 100])
  a.set_xlabel("pose-only AR", labelpad=1); a.set_ylabel("LaRC AR", labelpad=1)
  a.text(97, 34, f"{better}/{len(objs)} objects\nabove diagonal", ha="right", va="bottom", fontsize=5.4)
  p3(0.68, "(h) per-object AR")
  # (i) symmetric objects: the 16 samples spread over the symmetry orbit, yet each is correct
  rows_, obj_, Rg_, R_, _ = _samples()
  cls = {int(ob): ("continuous" if "symmetries_continuous" in info[str(ob)] else
                   ("discrete" if info[str(ob)].get("symmetries_discrete") else "none")) for ob in np.unique(obj_)}
  raw, sym, cl = [], [], []
  for i in range(len(rows_)):
    M = np.einsum("aji,bjk->abik", R_[i], R_[i])
    ang = np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    raw.append(np.median(ang[np.triu_indices(16, 1)])); sym.append(np.median(rot_err(Rg_[i], R_[i], int(obj_[i]))))
    cl.append(cls[int(obj_[i])])
  raw, sym, cl = np.array(raw), np.array(sym), np.array(cl)
  a = f3.add_axes([0.32, 0.085, 0.64, 0.16]); ks = ["none", "discrete", "continuous"]
  a.bar(np.arange(3) - 0.2, [np.median(raw[cl == k_]) for k_ in ks], 0.4, color=C["sym"], label="spread of 16")
  a.bar(np.arange(3) + 0.2, [np.median(sym[cl == k_]) for k_ in ks], 0.4, color=C["larc"], label="error (sym.)")
  a.set_yscale("log"); a.set_yticks([1, 10, 100]); a.set_yticklabels(["1", "10", "100"]); a.minorticks_off()
  a.set_ylim(0.8, 1500)
  a.set_xticks(range(3)); a.set_xticklabels(["none", "discr.", "cont."], fontsize=5.4); a.set_xlabel("object symmetry", labelpad=1)
  a.set_ylabel("deg (median)", labelpad=1); a.legend(loc="upper left", fontsize=4.6, handlelength=0.8, ncol=2, columnspacing=0.4)
  p3(0.315, "(i) symmetries")
  root.legend(*LEG, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.045), fontsize=6, handlelength=1.6)
  root.savefig(f"{OUT}/fig_teaser.pdf", bbox_inches="tight", pad_inches=0.02)
  root.savefig(f"{OUT}/fig_teaser.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
  plt.close(root)
  print("saved fig_teaser; per-object better", better, "spread", [np.median(raw[cl == k_]).round(1) for k_ in ks],
        "sym err", [np.median(sym[cl == k_]).round(2) for k_ in ks])


if __name__ == "__main__":
  which = sys.argv[1:] or ["render_compare", "in_loop", "trajectories", "energy", "geometry", "accuracy", "per_object",
                           "selection", "summary", "views", "occlusion", "population", "gallery", "robustness", "spread",
                           "failures", "mask_bias", "teaser"]
  for w in which:
    globals()[f"fig_{w}"]()
