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


if __name__ == "__main__":
  which = sys.argv[1:] or ["render_compare", "in_loop", "trajectories", "energy", "geometry", "accuracy", "per_object",
                           "selection", "summary"]
  for w in which:
    globals()[f"fig_{w}"]()
