"""Does the field predict unseen visual consequences? (read-only use of trained fields; T-LESS test set; no tuning)

  crossview  pairs of test crops of the same object in the same scene, different images (camera moved): the field
             renders the object at image B's pose, in B's crop camera, with the appearance code of image A, and is
             scored against B's real tokens (fraction of variance unexplained on B's visible-object patches).
             Controls: copy A's tokens (nothing changes); A's mean token; the field at A's pose in B's camera (motion
             ignored); the nearest training view to B's pose, appearance-matched to A (a view-based predictor with the
             same information); reference: the field with B's own appearance. Binned by the symmetry-aware rotation
             between A and B and by the crop-scale change. Frozen DINOv2 space (frozen field) and joint space.
  geometry   the field's occupancy (alpha = 1 - exp(-sigma * delta) at the renderer's sample spacing, > 0.5) on a 64^3
             grid in the object frame vs the solid CAD model (ray parity), all 30 objects: volumetric IoU, silhouette
             IoU and depth error of the six axis views, symmetric surface Chamfer distance (% of the diameter).
             Reference: the CAD's own filled bounding box. CAD is used for evaluation only.

Run: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false BOP_DS=tless PYTHONPATH=LARp/src:LARp/lib/bop_toolkit \
       python wm_world.py [crossview] [geometry] [--field DIR --tag NAME]
"""
import argparse
import json
import os
import sys
import types

for _m in ("bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from scipy.spatial.transform import Rotation

from bop_toolkit_lib import inout
from liepose.exp.cmp.sp_field_dit import FtFieldDenoiser
from liepose.exp.jepa_probe.tless_field import TlessField, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats

OUT = "/media/kavinder/hdd3/larc_figs/data"
DD = "/media/kavinder/hdd2/liepose_data"
AB = "/media/kavinder/hdd3/larp_abl"
TL = f"{DD}/bop_datasets/tless"
info = json.load(open(f"{TL}/models_eval/models_info.json"))
q2m = lambda q: Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()  # noqa: E731


def restore(path):
  c = ocp.PyTreeCheckpointer(); m = c.metadata(path)
  tree = m.item_metadata.tree if hasattr(m, "item_metadata") else m
  sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  return c.restore(path, restore_args=jax.tree_util.tree_map(lambda x: ocp.ArrayRestoreArgs(sharding=sh), tree))["params_ema"]


def rot_dist(Ra, Rb, o):
  """symmetry-aware angle (deg) between paired rotations (N,3,3)"""
  v = info[str(o)]
  if "symmetries_continuous" in v:
    ax = np.asarray(v["symmetries_continuous"][0]["axis"], float)
    cos = (np.einsum("nij,j->ni", Ra, ax) * np.einsum("nij,j->ni", Rb, ax)).sum(-1)
    e = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return np.minimum(e, 180 - e) if v.get("symmetries_discrete") else e
  Ss = [np.eye(3)] + [np.asarray(s, np.float64).reshape(4, 4)[:3, :3] for s in v.get("symmetries_discrete", [])]
  return np.min(np.stack([np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Ra @ S, Rb) - 1) / 2, -1, 1))) for S in Ss]), 0)


p = argparse.ArgumentParser()
p.add_argument("what", nargs="*", default=["crossview", "geometry"])
p.add_argument("--field", default=f"{AB}/probe/tl_field_v1")
p.add_argument("--tag", default="frozen")
a = p.parse_args()

field, fp_frozen = load_tless_field(a.field)
summ = {}

# ================================================================== cross-view prediction
if "crossview" in a.what:
  tm = np.load(f"{DD}/tless_tokens/test_meta.npz"); N = len(tm["object_id"])
  obj = tm["object_id"].astype(int); R = q2m(tm["rotation_allo"]).astype(np.float32); t = tm["translation_scaled"].astype(np.float32)
  k = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
  fg = np.load(f"{DD}/tless_tokens/test_fg.npy").astype(np.float32)
  dzi = np.load(f"{DD}/tless_crops_meta/test/shard_0000.npz")["dzi_size"].astype(np.float64)
  sig = np.load(f"{OUT}/test_scene_im_gt.npy")  # (scene, im, gt_id, match px) per test crop
  rng = np.random.default_rng(0)
  A, Bi = [], []
  for key in np.unique(sig[:, 0] * 100 + obj):
    rows = np.where(sig[:, 0] * 100 + obj == key)[0]
    for b in rows:
      cand = rows[sig[rows, 1] != sig[b, 1]]  # another image of the scene
      for a_ in rng.choice(cand, min(4, len(cand)), replace=False) if len(cand) else []:
        A.append(a_); Bi.append(b)
  A, Bi = np.array(A), np.array(Bi)
  drot = np.concatenate([rot_dist(R[A[obj[Bi] == o]].astype(np.float64), R[Bi[obj[Bi] == o]].astype(np.float64), o) for o in np.unique(obj[Bi])])
  order = np.concatenate([np.where(obj[Bi] == o)[0] for o in np.unique(obj[Bi])])
  dr = np.zeros(len(A)); dr[order] = drot
  dscale = np.abs(np.log(dzi[Bi] / dzi[A]))
  print("pairs", len(A), "rotation between views pct 50/90:", np.percentile(dr, [50, 90]).round(1), flush=True)
  spaces = {}
  z_f = np.asarray(np.load(f"{DD}/tless_tokens/test_tokens.npy", mmap_mode="r"), np.float32)
  nn = np.load(f"{OUT}/wm_diag_nn.npz")["nn"]
  ztr = np.load(f"{DD}/tless_tokens/train_tokens.npy", mmap_mode="r"); u = np.unique(nn)
  zn_f = np.asarray(ztr[u], np.float32)[np.searchsorted(u, nn)]
  spaces["frozen"] = (z_f, fp_frozen, zn_f)
  if a.tag == "frozen":  # joint model: its EMA-encoder space and its jointly trained field
    pj = restore(f"{AB}/a1_ckpt/checkpoint_100001")
    fp_joint = {"params": pj["field"]}; pj = {"params": {kk: v for kk, v in pj.items() if kk != "field"}}
    m = FtFieldDenoiser(mode="field_full_lat", dim=256, depth=4, use_k=False)
    enc = jax.jit(lambda im: m.apply(pj, im, method=FtFieldDenoiser.encode)[0])
    sh = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
    z_j = np.concatenate([np.asarray(enc(jnp.asarray(sh["image"][s:s + 64].astype(np.float32) * sh["mask"][s:s + 64, ..., None])))
                          for s in range(0, N, 64)])
    spaces["joint"] = (z_j, fp_joint, None)

  for sp, (z, fp, zn) in spaces.items():
    rend = jax.jit(lambda o, R_, t_, k_, zz: field.apply(fp, o, R_, t_, k_, app_stats(zz)))
    run = lambda Rs, zs: np.concatenate([np.asarray(rend(jnp.asarray(obj[Bi][s:s + 128]), Rs[s:s + 128], t[Bi][s:s + 128], k[Bi][s:s + 128],  # noqa
                                                          zs[s:s + 128])) for s in range(0, len(A), 128)])
    w = fg[Bi]; zB = z[Bi]
    mu = (fg[..., None] * z).sum((0, 1)) / fg.sum()
    var = (w * ((zB - mu) ** 2).sum(-1)).sum(1) + 1e-8
    fvu = lambda P: (w * ((P - zB) ** 2).sum(-1)).sum(1) / var  # noqa: E731
    E = {"field_A_appearance": fvu(run(R[Bi], z[A])), "field_B_appearance": fvu(run(R[Bi], z[Bi])),
         "field_at_A_pose": fvu(run(R[A], z[A])), "copy_A": fvu(z[A]),
         "mean_token_A": fvu(np.broadcast_to(z[A].mean(1, keepdims=True), zB.shape))}
    if zn is not None:
      zb = zn[Bi]
      E["nearest_view_matched_to_A"] = fvu((zb - zb.mean(1, keepdims=True)) / (zb.std(1, keepdims=True) + 1e-6) * z[A].std(1, keepdims=True)
                                           + z[A].mean(1, keepdims=True))
    ok = w.sum(1) > 2
    res = {"n_pairs": int(ok.sum()), "median": {kk: float(np.median(v[ok])) for kk, v in E.items()},
           "field_better_than": {kk: float((E["field_A_appearance"][ok] < v[ok]).mean()) for kk, v in E.items() if kk != "field_A_appearance"},
           "by_rotation": {}, "by_scale_change": {}}
    for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 90), (90, 181)):
      s = ok & (dr >= lo) & (dr < hi)
      if s.sum():
        res["by_rotation"][f"{lo}-{hi}"] = {"n": int(s.sum()), **{kk: float(np.median(v[s])) for kk, v in E.items()}}
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 9)):
      s = ok & (dscale >= lo) & (dscale < hi)
      if s.sum():
        res["by_scale_change"][f"{lo}-{hi}"] = {"n": int(s.sum()), **{kk: float(np.median(v[s])) for kk, v in E.items()}}
    summ[f"crossview_{sp}"] = res
    print(sp, json.dumps(res, indent=1), flush=True)
    np.savez(f"{OUT}/wm_world_crossview_{a.tag}_{sp}.npz", A=A, B=Bi, drot=dr, dscale=dscale, **E)


# ================================================================== geometry vs CAD
def hits(V, F, ax, n, lim):
  """intersections of rays parallel to axis ax through an n x n grid of columns over [-lim, lim]^2 with the mesh:
  list per column of the coordinates along ax"""
  o1, o2 = [i for i in range(3) if i != ax]
  T = V[F]  # (F, 3, 3)
  g = (np.arange(n) + 0.5) / n * 2 * lim - lim
  P1, P2 = np.meshgrid(g, g, indexing="ij"); P = np.stack([P1.ravel(), P2.ravel()], -1)  # (C, 2)
  out = [[] for _ in range(len(P))]
  for s in range(0, len(T), 2000):
    t_ = T[s:s + 2000]
    a2, b2, c2 = t_[:, 0, [o1, o2]], t_[:, 1, [o1, o2]], t_[:, 2, [o1, o2]]
    v0, v1 = b2 - a2, c2 - a2
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    good = np.abs(den) > 1e-12
    d = P[:, None, :] - a2[None]  # (C, f, 2)
    l1 = (d[..., 0] * v1[None, :, 1] - d[..., 1] * v1[None, :, 0]) / np.where(good, den, 1)[None]
    l2 = (v0[None, :, 0] * d[..., 1] - v0[None, :, 1] * d[..., 0]) / np.where(good, den, 1)[None]
    inside = good[None] & (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
    ci, fi = np.nonzero(inside)
    za = t_[fi, 0, ax] + l1[ci, fi] * (t_[fi, 1, ax] - t_[fi, 0, ax]) + l2[ci, fi] * (t_[fi, 2, ax] - t_[fi, 0, ax])
    for c, zz in zip(ci, za):
      out[c].append(zz)
  return out


if "geometry" in a.what:
  from liepose.exp.jepa_probe.probe_tless import load_symmetries  # noqa: F401  (import check)
  n = 64
  g = (np.arange(n) + 0.5) / n * 2 - 1
  G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(1, -1, 3).astype(np.float32)
  fields = {a.tag: fp_frozen}
  if a.tag == "frozen":
    fields["joint"] = {"params": restore(f"{AB}/a1_ckpt/checkpoint_100001")["field"]}
  geo = {nm: {} for nm in list(fields) + ["bbox_fill"]}
  for o in range(1, 31):
    r = float(field.radii[o - 1]); diam = info[str(o)]["diameter"] / 1000.0
    mm = inout.load_ply(f"{TL}/models_eval/obj_{o:06d}.ply")
    V, F = mm["pts"] / 1000.0, np.asarray(mm["faces"]).astype(int)
    # CAD solid by parity along z, and first hits along each axis (both directions)
    hz = hits(V, F, 2, n, r)
    zc = g * r
    cad = np.zeros((n, n, n), bool)
    for c, zz in enumerate(hz):
      if zz:
        zz = np.sort(zz); cad[c // n, c % n] = (np.searchsorted(zz, zc) % 2) == 1
    cad_d = {}
    for ax, hl in ((2, hz), (1, hits(V, F, 1, n, r)), (0, hits(V, F, 0, n, r))):
      for sgn in (1, -1):
        dmap = np.full(n * n, np.nan)
        for c, zz in enumerate(hl):
          if zz:
            dmap[c] = (r - max(zz)) if sgn == 1 else (min(zz) + r)  # depth from the + side / - side of the box
        cad_d[(ax, sgn)] = dmap.reshape(n, n)
    vs = (r * 2 / n)
    surf_cad = V[np.random.default_rng(0).choice(len(V), min(20000, len(V)), replace=False)]
    lo, hi = V.min(0), V.max(0)
    X = G[0].reshape(n, n, n, 3) * r
    bbox = np.all((X >= lo) & (X <= hi), -1)
    for nm, fp in fields.items():
      sg, _ = field.apply(fp, jnp.asarray(G), jnp.array([o - 1]), method=TlessField.point)
      sg = np.asarray(sg[0]).reshape(n, n, n)
      # the density is a soft volume (fog over the whole box), so a per-voxel threshold is meaningless: measure
      # (i) whether density localises the object (AUROC inside vs outside the CAD solid, threshold-free),
      # (ii) shape at matched volume (top-k voxels, k = CAD volume; optimistic: borrows the CAD volume),
      # (iii) the rendered silhouettes at 50% transmittance along the six axis directions (what the renderer sees)
      inside, outside = sg[cad], sg[~cad]
      from sklearn.metrics import roc_auc_score
      auroc = float(roc_auc_score(np.r_[np.ones(len(inside)), np.zeros(len(outside))], np.r_[inside, outside])) if cad.any() else np.nan
      kv = int(cad.sum())
      occ = np.zeros(sg.size, bool); occ[np.argsort(sg.ravel())[::-1][:kv]] = True; occ = occ.reshape(sg.shape)
      iou = float((occ & cad).sum() / max((occ | cad).sum(), 1))
      sil, dep, opq, cadf = [], [], [], []
      for (ax, sgn), dm in cad_d.items():
        a3 = np.moveaxis(1 - np.exp(-sg * vs), ax, -1)  # opacity per voxel along the axis
        if sgn == 1:
          a3 = a3[..., ::-1]
        T = np.cumprod(1 - a3, -1)
        hit = T < 0.5
        first = np.where(hit.any(-1), hit.argmax(-1), -1)
        fd = np.where(first >= 0, (first + 0.5) * vs, np.nan)
        m_f, m_c = ~np.isnan(fd), ~np.isnan(dm)
        sil.append((m_f & m_c).sum() / max((m_f | m_c).sum(), 1)); opq.append(m_f.mean()); cadf.append(m_c.mean())
        both = m_f & m_c
        dep.append(np.mean(np.abs(fd[both] - dm[both])) / diam if both.any() else np.nan)
      b = occ & ~(np.roll(occ, 1, 0) & np.roll(occ, -1, 0) & np.roll(occ, 1, 1) & np.roll(occ, -1, 1) & np.roll(occ, 1, 2) & np.roll(occ, -1, 2))
      Ps = X[b]
      if len(Ps):
        from scipy.spatial import cKDTree
        ch = float((cKDTree(surf_cad).query(Ps)[0].mean() + cKDTree(Ps).query(surf_cad)[0].mean()) / 2 / diam)
      else:
        ch = np.nan
      geo[nm][o] = dict(auroc_inside=auroc, iou_matched_volume=iou, sil_iou=float(np.mean(sil)), opaque_frac=float(np.mean(opq)),
                        cad_frac=float(np.mean(cadf)), depth_err=float(np.nanmean(dep)), chamfer_matched_volume=ch)
    geo["bbox_fill"][o] = dict(iou_matched_volume=float((bbox & cad).sum() / max((bbox | cad).sum(), 1)))
    print("geometry obj", o, {nm: {kk: round(v, 3) for kk, v in geo[nm][o].items()} for nm in geo}, flush=True)
  summ["geometry"] = {nm: {kk: float(np.nanmedian([geo[nm][o][kk] for o in geo[nm]])) for kk in geo[nm][1]} for nm in geo}
  summ["geometry_per_object"] = {nm: {str(o): v for o, v in d.items()} for nm, d in geo.items()}
  print(json.dumps(summ["geometry"], indent=1), flush=True)

json.dump(summ, open(f"{OUT}/wm_world_{a.tag}_{'_'.join(a.what)}.json", "w"), indent=1)
print("saved")
