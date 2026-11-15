"""exp1 fairness check (frozen space): field vs nearest training view, scored only on patches visible in both crops"""
import sys, types, json
for _m in ("bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))
import numpy as np, jax, jax.numpy as jnp
from scipy.spatial.transform import Rotation
from liepose.exp.jepa_probe.tless_field import load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats
DD = "/media/kavinder/hdd2/liepose_data"; OUT = "/media/kavinder/hdd3/larc_figs/data"
q2m = lambda q: Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()
tm = np.load(f"{DD}/tless_tokens/test_meta.npz"); N = len(tm["object_id"])
z = np.asarray(np.load(f"{DD}/tless_tokens/test_tokens.npy", mmap_mode="r"), np.float32)
fg = np.load(f"{DD}/tless_tokens/test_fg.npy").astype(np.float32)
k = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
nn = np.load(f"{OUT}/wm_diag_nn.npz")["nn"]; dnn = np.load(f"{OUT}/wm_diag_nn.npz")["dnn"]
ztr = np.load(f"{DD}/tless_tokens/train_tokens.npy", mmap_mode="r"); fgtr = np.load(f"{DD}/tless_tokens/train_fg.npy", mmap_mode="r")
u = np.unique(nn); pos = np.searchsorted(u, nn)
zn = np.asarray(ztr[u], np.float32)[pos]; fgn = np.asarray(fgtr[u], np.float32)[pos]
field, fp = load_tless_field("/media/kavinder/hdd3/larp_abl/probe/tl_field_v1")
r = jax.jit(lambda o, R, t, kk, zz: field.apply(fp, o, R, t, kk, app_stats(zz)))
R = q2m(tm["rotation_allo"]).astype(np.float32); t = tm["translation_scaled"].astype(np.float32); o = tm["object_id"]
zh = np.concatenate([np.asarray(r(jnp.asarray(o[s:s+64]), R[s:s+64], t[s:s+64], k[s:s+64], z[s:s+64])) for s in range(0, N, 64)])
mu = (fg[..., None] * z).sum((0, 1)) / fg.sum()
# controls: (a) this crop's own mean token at every patch (what the appearance statistics alone give);
# (b) the nearest view with per-channel mean and std matched to this crop (appearance-corrected lookup)
zm = np.broadcast_to(z.mean(1, keepdims=True), z.shape)
zn_m = (zn - zn.mean(1, keepdims=True)) / (zn.std(1, keepdims=True) + 1e-6) * z.std(1, keepdims=True) + z.mean(1, keepdims=True)
out = {}
for nm, w in (("test_fg", fg), ("covisible", fg * fgn)):
  var = (w * ((z - mu) ** 2).sum(-1)).sum(1) + 1e-8
  ok = w.sum(1) > 2
  a = (w * ((zh - z) ** 2).sum(-1)).sum(1)[ok] / var[ok]; b = (w * ((zn - z) ** 2).sum(-1)).sum(1)[ok] / var[ok]
  c = (w * ((zm - z) ** 2).sum(-1)).sum(1)[ok] / var[ok]; d = (w * ((zn_m - z) ** 2).sum(-1)).sum(1)[ok] / var[ok]
  out[nm] = dict(n=int(ok.sum()), field=float(np.median(a)), nn=float(np.median(b)), field_better=float((a < b).mean()),
                 own_mean_token=float(np.median(c)), nn_appearance_matched=float(np.median(d)),
                 field_better_than_own_mean=float((a < c).mean()), field_better_than_nn_matched=float((a < d).mean()),
                 by_dist={f"{lo}-{hi}": [float(np.median(a[(dnn[ok] >= lo) & (dnn[ok] < hi)])), float(np.median(b[(dnn[ok] >= lo) & (dnn[ok] < hi)]))]
                          for lo, hi in ((0, 2), (2, 5), (5, 20))})
print(json.dumps(out, indent=1)); json.dump(out, open(f"{OUT}/wm_diag_covis.json", "w"), indent=1)
