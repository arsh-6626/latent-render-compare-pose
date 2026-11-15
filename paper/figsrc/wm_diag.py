"""World-model diagnostics (read-only use of trained checkpoints; T-LESS test set, GT crops; no tuning).

  exp1  generalisation to unseen poses: token prediction error at the GT pose, the field vs the nearest training view
        (symmetry-aware nearest pose among the 180k training crops of the object), binned by that distance.
        Two token spaces: frozen DINOv2 (frozen field tl_field_v1) and the joint model's EMA encoder (joint field).
        Error = fraction of variance unexplained on visible-object patches (fg-weighted), so spaces are comparable.
  exp2  amodal prediction and surprise, 1500 seeded test crops: patches labelled from the visible mask and the projected
        CAD silhouette at the GT pose (labels for evaluation only). Opacity on occluded-object vs background patches
        (frozen, joint, mask-free models); residual and WM weights on visible vs occluded patches (mask-free model on the
        unmasked crop: the only model that is trained to see occluders).
  exp5  per-object: world-model quality (exp1 error, exp2 occluder AUROC) vs the per-object AR gain of LaRC over control.

Run: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false BOP_DS=tless DATADIR=/media/kavinder/hdd2/liepose_data \
       PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python wm_diag.py      -> /media/kavinder/hdd3/larc_figs/data/wm_diag*.npz|json
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
import orbax.checkpoint as ocp
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from liepose.exp.cmp.sp_field_dit import FtFieldDenoiser
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.tless_field import TlessGeometry, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats

OUT = "/media/kavinder/hdd3/larc_figs/data"
DD = "/media/kavinder/hdd2/liepose_data"
AB = "/media/kavinder/hdd3/larp_abl"
TL = f"{DD}/bop_datasets/tless"
SH = 8192
info = json.load(open(f"{TL}/models_eval/models_info.json"))


def wxyz_to_mat(q):
  return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()


def restore(path):
  c = ocp.PyTreeCheckpointer()
  m = c.metadata(path)
  tree = m.item_metadata.tree if hasattr(m, "item_metadata") else m
  sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  return c.restore(path, restore_args=jax.tree_util.tree_map(lambda x: ocp.ArrayRestoreArgs(sharding=sh), tree))["params_ema"]


def sym_mats(o):
  return np.stack([np.eye(3)] + [np.asarray(s, np.float64).reshape(4, 4)[:3, :3] for s in info[str(o)].get("symmetries_discrete", [])])


def rot_dist(Rq, Rs, o):
  """symmetry-aware angle (deg) between Rq (3,3) and each of Rs (N,3,3); continuous symmetry -> angle between axes"""
  v = info[str(o)]
  if "symmetries_continuous" in v:
    ax = np.asarray(v["symmetries_continuous"][0]["axis"], float)
    e = np.degrees(np.arccos(np.clip(np.einsum("nij,j->ni", Rs, ax) @ (Rq @ ax), -1, 1)))
    return np.minimum(e, 180 - e) if v.get("symmetries_discrete") else e
  out = []
  for S in sym_mats(o):
    tr = np.einsum("ij,nij->n", Rq @ S, Rs)  # trace((Rq S)^T Rs)
    out.append(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
  return np.min(np.stack(out), 0)


# ------------------------------------------------------------------ data
tm = np.load(f"{DD}/tless_tokens/test_meta.npz")
sh0 = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
assert (sh0["index"] == tm["index"]).all()
test_img, test_mask = sh0["image"], sh0["mask"]
test_k = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
test_tok = np.load(f"{DD}/tless_tokens/test_tokens.npy", mmap_mode="r")
test_fg = np.load(f"{DD}/tless_tokens/test_fg.npy").astype(np.float32)
obj = tm["object_id"].astype(int)
R_te = wxyz_to_mat(tm["rotation_allo"]).astype(np.float32)
t_te = tm["translation_scaled"].astype(np.float32)
trm = np.load(f"{DD}/tless_tokens/train_meta.npz")
R_tr = wxyz_to_mat(trm["rotation_allo"])
obj_tr = trm["object_id"].astype(int)
train_tok = np.load(f"{DD}/tless_tokens/train_tokens.npy", mmap_mode="r")
N = len(obj)

# nearest training view (symmetry-aware rotation distance, same object)
f_nn = f"{OUT}/wm_diag_nn.npz"
if not os.path.exists(f_nn):
  nn, dnn = np.zeros(N, int), np.zeros(N)
  for o in np.unique(obj):
    rows_tr = np.where(obj_tr == o)[0]
    for i in np.where(obj == o)[0]:
      d = rot_dist(R_te[i].astype(np.float64), R_tr[rows_tr], o)
      j = int(np.argmin(d)); nn[i], dnn[i] = rows_tr[j], d[j]
    print("nn obj", o, flush=True)
  np.savez(f_nn, nn=nn, dnn=dnn)
NN = np.load(f_nn); nn, dnn = NN["nn"], NN["dnn"]
print("nearest-view distance percentiles (deg) 50/90/99:", np.percentile(dnn, [50, 90, 99]).round(2), flush=True)

# ------------------------------------------------------------------ models
field, fp_frozen = load_tless_field(f"{AB}/probe/tl_field_v1")
fieldw, _ = load_tless_field("/media/kavinder/hdd3/larp_jdet/probe/tl_field_det")
pj = restore(f"{AB}/a1_ckpt/checkpoint_100001")
fp_joint = {"params": pj["field"]}; pj = {"params": {k: v for k, v in pj.items() if k != "field"}}
pw = restore("/media/kavinder/hdd3/larc_figs/wmw_ckpt/checkpoint_100000")
fp_wmw = {"params": pw["field"]}; pw = {"params": {k: v for k, v in pw.items() if k != "field"}}
m = FtFieldDenoiser(mode="field_full_lat", dim=256, depth=4, use_k=False)
enc_joint = jax.jit(lambda im: m.apply(pj, im, method=FtFieldDenoiser.encode)[0])
enc_wmw = jax.jit(lambda im: m.apply(pw, im, method=FtFieldDenoiser.encode)[0])


def make_render(fm, fp):
  @jax.jit
  def r(o, R, t, k, z):
    zh, op, _ = fm.apply(fp, o, R, t, k, app_stats(z), with_opacity=True, with_latent=True)
    return zh, op
  return r


rend = {"frozen": make_render(field, fp_frozen), "joint": make_render(field, fp_joint), "wmw": make_render(fieldw, fp_wmw)}
B = 64


def batched(fn, n, *arrs):
  outs = []
  for s in range(0, n, B):
    outs.append(jax.device_get(fn(*[jnp.asarray(a[s:s + B]) for a in arrs])))
  return [np.concatenate([o[i] for o in outs]) for i in range(len(outs[0]))] if isinstance(outs[0], tuple) else np.concatenate(outs)


def load_train_crops(rows):
  """images and masks of training rows (only the shards that are needed)"""
  img = np.zeros((len(rows), 224, 224, 3), np.uint8); msk = np.zeros((len(rows), 224, 224), np.float32)
  order = np.argsort(rows)
  for s in np.unique(rows // SH):
    d = np.load(f"{DD}/tless_crops/train/shard_{s:04d}.npz")
    sel = order[(rows[order] // SH) == s]
    assert (d["index"][rows[sel] % SH] == trm["index"][rows[sel]]).all()
    img[sel] = d["image"][rows[sel] % SH]; msk[sel] = d["mask"][rows[sel] % SH]
    print("train shard", s, flush=True)
  return img, msk


# ------------------------------------------------------------------ exp1
f1 = f"{OUT}/wm_diag_exp1.npz"
if not os.path.exists(f1):
  E = {}
  idx = np.arange(N)
  ko = lambda a: a  # noqa
  # frozen space: precomputed tokens (frozen DINOv2 of the masked crop)
  z_f = np.asarray(test_tok, np.float32)
  zh_f, op_f = batched(rend["frozen"], N, obj, R_te, t_te, test_k, z_f)
  zl_f = np.asarray(train_tok[np.sort(np.unique(nn))], np.float32)
  pos = {r: i for i, r in enumerate(np.sort(np.unique(nn)))}
  zn_f = zl_f[[pos[r] for r in nn]]
  # joint space: EMA encoder of the masked crops (test and the nearest training crops)
  im_te = (test_img.astype(np.float32) * test_mask[..., None])
  z_j = batched(enc_joint, N, im_te)
  zh_j, op_j = batched(rend["joint"], N, obj, R_te, t_te, test_k, z_j)
  rows_u = np.sort(np.unique(nn)); ti, tmk = load_train_crops(rows_u)
  zl_j = batched(enc_joint, len(rows_u), ti.astype(np.float32) * tmk[..., None])
  zn_j = zl_j[[pos[r] for r in nn]]
  for sp, z, zh, zn in (("frozen", z_f, zh_f, zn_f), ("joint", z_j, zh_j, zn_j)):
    w = test_fg
    mu = (w[..., None] * z).sum((0, 1)) / w.sum()
    var = (w * ((z - mu) ** 2).sum(-1)).sum(1)  # per crop, fg-weighted
    E[f"{sp}_fvu_field"] = (w * ((zh - z) ** 2).sum(-1)).sum(1) / var
    E[f"{sp}_fvu_nn"] = (w * ((zn - z) ** 2).sum(-1)).sum(1) / var
  E["op_frozen"], E["op_joint"] = op_f, op_j
  np.savez(f1, dnn=dnn, **E)
E = dict(np.load(f1))
bins = [0, 1, 2, 5, 10, 20, 180]
summ = {"nn_dist_pct_50_90_99": np.percentile(dnn, [50, 90, 99]).round(2).tolist(), "exp1": {}}
for sp in ("frozen", "joint"):
  rows = []
  for lo, hi in zip(bins[:-1], bins[1:]):
    s = (dnn >= lo) & (dnn < hi)
    if s.sum():
      rows.append(dict(bin=f"{lo}-{hi}", n=int(s.sum()), field=float(np.median(E[f"{sp}_fvu_field"][s])),
                       nn=float(np.median(E[f"{sp}_fvu_nn"][s]))))
  summ["exp1"][sp] = dict(by_bin=rows, all_field=float(np.median(E[f"{sp}_fvu_field"])), all_nn=float(np.median(E[f"{sp}_fvu_nn"])),
                          field_better=float((E[f"{sp}_fvu_field"] < E[f"{sp}_fvu_nn"]).mean()),
                          spearman_field_vs_dist=float(spearmanr(dnn, E[f"{sp}_fvu_field"])[0]),
                          spearman_nn_vs_dist=float(spearmanr(dnn, E[f"{sp}_fvu_nn"])[0]))
print(json.dumps(summ["exp1"], indent=1), flush=True)

# ------------------------------------------------------------------ exp2
geo = TlessGeometry()
_mesh = {}
from PIL import Image, ImageDraw  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402
from bop_toolkit_lib import inout  # noqa: E402


def silhouette(o, R_allo, t_s, k, res=224):
  if o not in _mesh:
    mm = inout.load_ply(f"{TL}/models_eval/obj_{o:06d}.ply")
    _mesh[o] = (mm["pts"] / 1000.0, np.asarray(mm["faces"]).astype(int))
  v, f = _mesh[o]
  R, t = geo.cam_pose(jnp.asarray(R_allo)[None], jnp.asarray(t_s)[None], jnp.asarray(k)[None])
  Pc = v @ np.asarray(R[0]).T + np.asarray(t[0])
  u = Pc[:, :2] / Pc[:, 2:] * k[6:8] + k[8:10]
  u = (u - k[1:3]) * k[0]
  im = Image.new("L", (res, res), 0); dr = ImageDraw.Draw(im)
  for tri in f:
    dr.polygon([tuple(u[i]) for i in tri], fill=255)
  return np.asarray(im) > 0


pool8 = lambda a: a.reshape(8, 28, 8, 28).mean((1, 3))  # noqa: E731
f2 = f"{OUT}/wm_diag_exp2.npz"
if not os.path.exists(f2):
  rows = np.sort(np.random.default_rng(7).choice(N, 1500, replace=False))
  sil8 = np.stack([pool8(silhouette(int(obj[r]), R_te[r], t_te[r], test_k[r]).astype(np.float32)) for r in rows])
  vis8 = np.stack([pool8((test_mask[r] > 0).astype(np.float32)) for r in rows])
  z_w = batched(enc_wmw, len(rows), test_img[rows].astype(np.float32))  # unmasked crop
  zh_w, op_w = batched(rend["wmw"], len(rows), obj[rows], R_te[rows], t_te[rows], test_k[rows], z_w)
  r2 = ((zh_w - z_w) ** 2).mean(-1)
  k2 = np.median(r2, -1, keepdims=True) + 1e-8
  w_w = FG_FLOOR + np.clip(op_w, 0, 1) * k2 / (k2 + r2)  # = tless_field_dit._weights
  np.savez(f2, rows=rows, sil8=sil8.reshape(-1, 64), vis8=vis8.reshape(-1, 64), op_wmw=op_w, r2_wmw=r2, w_wmw=w_w,
           op_frozen=E["op_frozen"][rows], op_joint=E["op_joint"][rows])
X = dict(np.load(f2))
sil, vis = X["sil8"], X["vis8"]
VIS = vis >= 0.5
OCC = (sil >= 0.5) & (vis < 0.1)
BG = (sil < 0.05) & (vis < 0.05)
auc = lambda pos, neg: float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))  # noqa: E731
e2 = {"patches": dict(visible=int(VIS.sum()), occluded=int(OCC.sum()), background=int(BG.sum()),
                      crops_with_occluded=int(OCC.any(1).sum()))}
for nm in ("frozen", "joint", "wmw"):
  op = X[f"op_{nm}"]
  e2[f"opacity_{nm}"] = dict(visible=float(op[VIS].mean()), occluded=float(op[OCC].mean()), background=float(op[BG].mean()),
                             auroc_occluded_vs_background=auc(op[OCC], op[BG]),
                             iou_full_silhouette=float(((op > 0.5) & (sil >= 0.5)).sum() / ((op > 0.5) | (sil >= 0.5)).sum()),
                             iou_visible_mask=float(((op > 0.5) & VIS).sum() / ((op > 0.5) | VIS).sum()))
r2, ww = X["r2_wmw"], X["w_wmw"]
r2n = r2 / np.median(r2, -1, keepdims=True)
e2["surprise_wmw"] = dict(residual_auroc_occluded_vs_visible=auc(r2n[OCC], r2n[VIS]),
                          weight_visible=float(ww[VIS].mean()), weight_occluded=float(ww[OCC].mean()),
                          weight_background=float(ww[BG].mean()),
                          weight_auroc_visible_vs_rest=auc(ww[VIS], ww[OCC | BG]),
                          weight_auroc_visible_vs_occluded=auc(ww[VIS], ww[OCC]))
summ["exp2"] = e2
print(json.dumps(e2, indent=1), flush=True)

# ------------------------------------------------------------------ exp5: per-object WM quality vs AR gain
sys.path.insert(0, os.path.dirname(__file__))
import figs  # noqa: E402

th_s, th_p = np.arange(0.05, 0.51, 0.05), np.arange(5, 51, 5)
per = {}
for name in ("field_nowm_ft", "joint_n1"):
  ms, mp = figs.load_err(name, "mssd"), figs.load_err(name, "mspd"); acc = {}
  for kk in ms:
    acc.setdefault(kk[2], []).append(((ms[kk] / figs.DIAM[kk[2]] < th_s).mean() + (mp.get(kk, 1e9) < th_p).mean()) / 2)
  per[name] = {o: np.mean(v) * 100 for o, v in acc.items()}
objs = sorted(per["joint_n1"])
gain = np.array([per["joint_n1"][o] - per["field_nowm_ft"][o] for o in objs])
fvu = np.array([np.median(E["joint_fvu_field"][obj == o]) for o in objs])
adv = np.array([np.median(E["joint_fvu_nn"][obj == o] - E["joint_fvu_field"][obj == o]) for o in objs])
ro = X["rows"]
occ_auc = []
for o in objs:
  s = obj[ro] == o
  O_, V_ = OCC[s], VIS[s]
  occ_auc.append(auc(r2n[s][O_], r2n[s][V_]) if O_.sum() >= 5 and V_.sum() >= 5 else np.nan)
occ_auc = np.array(occ_auc)
ok = ~np.isnan(occ_auc)
summ["exp5"] = dict(objects=objs, ar_gain=gain.round(2).tolist(), joint_fvu=fvu.round(4).tolist(), field_adv_over_nn=adv.round(4).tolist(),
                    occ_auroc=[None if np.isnan(a) else round(float(a), 3) for a in occ_auc],
                    spearman_gain_vs_fvu=list(map(float, spearmanr(gain, fvu))),
                    spearman_gain_vs_field_adv=list(map(float, spearmanr(gain, adv))),
                    spearman_gain_vs_occ_auroc=list(map(float, spearmanr(gain[ok], occ_auc[ok]))))
print(json.dumps({k: v for k, v in summ["exp5"].items() if k.startswith("spearman")}, indent=1), flush=True)
json.dump(summ, open(f"{OUT}/wm_diag.json", "w"), indent=1)
print("saved", f"{OUT}/wm_diag.json")
