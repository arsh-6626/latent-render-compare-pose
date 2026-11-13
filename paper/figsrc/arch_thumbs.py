"""Thumbnails from real inference for the draw.io architecture diagrams (traj_4 = fig_in_loop's crop, T-LESS object 23;
the joint model's own field parameters). CPU only.
Run: JAX_PLATFORMS=cpu BOP_DS=tless DATADIR=... PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python arch_thumbs.py
-> LARp/paper/fig/arch/thumbs/*.png"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import figs  # noqa: E402  (style, silhouette, pca_basis, trajectories; no figure is drawn on import)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

OUT = "/home/kavinder/ARNABI_ARSH/LARp/paper/fig/arch/thumbs"
os.makedirs(OUT, exist_ok=True)
CI, STEPS, CH = 4, (0, 60, 99), 0
t = figs.traj[CI]
o = int(t["obj"])


def save_img(name, arr, cmap=None, vmin=None, vmax=None, contours=(), px=200):
  fig = plt.figure(figsize=(1, 1), dpi=px)
  a = fig.add_axes([0, 0, 1, 1]); a.axis("off")
  a.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
  for m, c, ls, lw in contours:
    figs.contour(a, m, c, lw, ls)
  if arr.shape[0] == 224:
    a.set_xlim(0, 223); a.set_ylim(223, 0)
  fig.savefig(f"{OUT}/{name}.png", dpi=px); plt.close(fig)


z, w = t["z_joint"], t["w"]
zh = [t["joint_zh0"][s].astype(np.float32) for s in STEPS]
col = figs.pca_basis(np.concatenate([z, t["gt_zh_joint"]] + zh))
s_gt = figs.silhouette(o, t["R_gt"], t["t_gt"], t["k"])
save_img("crop", t["img"])
save_img("crop_masked", (t["img"] * t["mask"][..., None]).astype(np.uint8))
save_img("tokens_obs", col(z))
res = [figs.residual(x, z, w) for x in zh]
vmax = np.percentile(np.concatenate([r.ravel() for r in res]), 99)
for s, x, r in zip(STEPS, zh, res):
  sil = figs.silhouette(o, t["joint_R"][s, CH], t["joint_T"][s, CH], t["k"])
  dim = (t["img"] * (0.35 + 0.65 * t["mask"][..., None])).astype(np.uint8)
  save_img(f"hyp_{s + 1}", dim, contours=[(s_gt, "w", "--", 0.6), (sil, figs.C["sym"], "-", 1.0)])
  save_img(f"render_{s + 1}", col(x))
  save_img(f"opacity_{s + 1}", t["joint_op0"][s].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
  save_img(f"residual_{s + 1}", r, cmap=figs.RES, vmin=0, vmax=vmax)
  print("step", s + 1, "rot err", round(float(t["joint_rerr"][s, CH]), 1))
# final poses of the 16 chains and the GT
dim = (t["img"] * (0.35 + 0.65 * t["mask"][..., None])).astype(np.uint8)
cs = [(figs.silhouette(o, t["joint_R"][-1, c], t["joint_T"][-1, c], t["k"]), figs.C["larc"], "-", 0.5) for c in range(16)]
save_img("final_16", dim, contours=cs + [(s_gt, "w", "--", 0.6)])
# noisy training poses (chains early in sampling = noisy hypotheses around nothing in particular)
cs = [(figs.silhouette(o, t["joint_R"][50, c], t["joint_T"][50, c], t["k"]), figs.C["sym"], "-", 0.6) for c in range(4)]
save_img("noisy_poses", dim, contours=cs + [(s_gt, "w", "--", 0.6)])
# render at the GT pose and the field-loss residual
save_img("render_gt", col(t["gt_zh_joint"]))
save_img("residual_gt", figs.residual(t["gt_zh_joint"], z, w), cmap=figs.RES, vmin=0, vmax=vmax)
save_img("opacity_gt", t["gt_op_joint"].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
save_img("weights", w.reshape(8, 8), cmap="gray", vmin=0, vmax=1.05)

# tri-planes of the object (joint field), PCA to RGB over the three planes together
import jax  # noqa: E402
c = ocp.PyTreeCheckpointer(); cp = "/media/kavinder/hdd3/larp_abl/a1_ckpt/checkpoint_100001"
m = c.metadata(cp); tree = m.item_metadata.tree if hasattr(m, "item_metadata") else m
sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
p = c.restore(cp, restore_args=jax.tree_util.tree_map(lambda x: ocp.ArrayRestoreArgs(sharding=sh), tree))["params_ema"]["field"]
P = {k: np.asarray(p[k][o - 1]) for k in ("plane_xy", "plane_xz", "plane_yz")}
X = np.concatenate([v.reshape(-1, 32) for v in P.values()])
mu = X.mean(0); _, _, Vt = np.linalg.svd(X[::7] - mu, full_matrices=False)
Y = (X - mu) @ Vt[:3].T; lo, hi = np.percentile(Y, 1, 0), np.percentile(Y, 99, 0)
for i, k in enumerate(P):
  rgb = np.clip((Y[i * 128 * 128:(i + 1) * 128 * 128] - lo) / (hi - lo), 0, 1).reshape(128, 128, 3)
  save_img(k, rgb)
# field depth along z (as fig_geometry)
d = np.load(f"{figs.D}/density.npz"); n = len(d["grid"]); sg = d[f"joint_{o}"][..., ::-1]
dz = 2.0 * float(d[f"r_{o}"]) / n
hit = np.cumsum(sg * dz, -1) >= 1.0
first = np.where(hit.any(-1), hit.argmax(-1), n)
depth = np.where(first < n, 1 - first / n, np.nan)
save_img("depth_z", np.nan_to_num(depth, nan=0.0).T[::-1], cmap="magma", vmin=0.2, vmax=1.0)
print("saved to", OUT)
