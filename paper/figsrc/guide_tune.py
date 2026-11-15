"""Energy-guidance ablation: choose the guidance weight on TRAINING crops (never test), for the control denoiser
(field_nowm_ft, trained encoder, 100k x 32) guided by the gradient of the Stage-1 field's energy in its comparison space.

Rule fixed before running: 300 training crops from the source images held out of the field fit (rng 1234, as the fit's
split; seed 5 for the draw), weights {0, 0.003, 0.01, 0.03, 0.1, 0.3, 1} (a first grid {0, 3, ..., 1000} already failed
at 3 on these training crops, so the grid was moved down, still on training crops only), criterion = mean of the share of crops with
symmetry-aware rotation error < 5 deg and translation error < 2 cm. The chosen weight is then run once on the BOP test set.

Run: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false BOP_DS=tless BOP_RENDERER=vispy DATADIR=... CACHE_ROOT_PATH=...
     PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python guide_tune.py EXP_PATH
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

from liepose.exp.cmp import tless_field_dit as tfd
from liepose.exp.cmp.tless_dit import mask_fg
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.tless_field import TlessGeometry
from liepose.utils.config import get_args, load_config_from_args

EXP = sys.argv[1]
DD = os.environ["DATADIR"]
info = json.load(open(f"{DD}/bop_datasets/tless/models_eval/models_info.json"))
q2m = lambda q: Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()  # noqa: E731


def rot_err(Rg, Rm, o):
  v = info[str(o)]
  if "symmetries_continuous" in v:
    ax = np.asarray(v["symmetries_continuous"][0]["axis"], float)
    e = np.degrees(np.arccos(np.clip((Rg @ ax) @ (Rm @ ax), -1, 1)))
    return min(e, 180 - e) if v.get("symmetries_discrete") else e
  Ss = [np.eye(3)] + [np.asarray(s, float).reshape(4, 4)[:3, :3] for s in v.get("symmetries_discrete", [])]
  return min(np.degrees(np.arccos(np.clip((np.trace((Rg @ S).T @ Rm) - 1) / 2, -1, 1))) for S in Ss)


sys.argv = [sys.argv[0], "--root_path", os.path.dirname(EXP), "den_mode=[field_nowm_ft]",
            f"field_dir=[/media/kavinder/hdd3/larp_abl/probe/tl_field_v1]", "train_steps=[100000]", "seed=[42]",
            "batch_size=[32]", "n_slices=[16]", "skip_train=[True]"]
a = get_args(exp_name="tless-field", desc="guidance tuning")
opt = next(iter(load_config_from_args(a, tfd.build_explorations()).expand()))
opt.pop("sample_options", None)
tb = tfd.Testbed(opt, EXP, setup=True)
assert int(tb.state.step) >= 100000, tb.state.step
from liepose.utils.exploration import Options  # noqa: E402
tb.opt = Options(tfd.SAMPLE)  # the test-time sampling options (as run_test)

trm = np.load(f"{DD}/tless_tokens/train_meta.npz")
grp = np.unique(trm["index"]); held = np.random.default_rng(1234).permutation(grp)[: int(0.03 * len(grp))]
cand = np.nonzero(np.isin(trm["index"], held))[0]
rows = np.sort(np.random.default_rng(5).choice(cand, 300, replace=False))
k = np.load(f"{DD}/tless_tokens/train_crop_meta.npy").astype(np.float32)[rows]
obj = trm["object_id"][rows].astype(np.int32); Rgt = q2m(trm["rotation_allo"][rows]); tgt = trm["translation_scaled"][rows].astype(np.float32)
img = np.zeros((len(rows), 224, 224, 3), np.float32); msk = np.zeros((len(rows), 224, 224), np.float32)
for s in np.unique(rows // 8192):
  d = np.load(f"{DD}/tless_crops/train/shard_{s:04d}.npz"); sel = np.nonzero(rows // 8192 == s)[0]
  assert (d["index"][rows[sel] % 8192] == trm["index"][rows[sel]]).all()
  img[sel] = d["image"][rows[sel] % 8192]; msk[sel] = d["mask"][rows[sel] % 8192] > 0
image = jnp.asarray(img * msk[..., None]); w = FG_FLOOR + mask_fg(jnp.asarray(msk))
geo = TlessGeometry(); poses = tb._poses_fn()
lm, Lie = tb.lie_metrics, tb.Lie
prior = jax.jit(jax.vmap(lambda kk: lm.as_repr(Lie.sample_uniform(kk), tb.a.repr_type)))
_, t_gt_cam = geo.cam_pose(jnp.asarray(Rgt, jnp.float32), jnp.asarray(tgt), jnp.asarray(k))
res = {}
for gw in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
  tb.a.guide_w = gw
  sample = tb.create_sampler(100)
  out = []
  for s in range(0, len(rows), 50):
    key = jax.random.PRNGKey(7 + s)
    k1, k2 = jax.random.split(key)
    rt = sample(k2, prior(jax.random.split(k1, len(rows[s:s + 50]))), obj[s:s + 50], None, w[s:s + 50], jnp.asarray(k[s:s + 50]), image[s:s + 50])
    out.append(np.asarray(rt))
  rt = jnp.asarray(np.concatenate(out))
  Rm, tm_ = poses(rt)
  _, t_cam = geo.cam_pose(Rm, tm_, jnp.asarray(k))
  re = np.array([rot_err(Rgt[i], np.asarray(Rm[i], float), obj[i]) for i in range(len(rows))])
  te = np.linalg.norm(np.asarray(t_cam) - np.asarray(t_gt_cam), axis=1) * 100  # cm
  res[gw] = dict(R5=float((re < 5).mean()), T2=float((te < 2).mean()), crit=float(((re < 5).mean() + (te < 2).mean()) / 2),
                 rot_med=float(np.median(re)), tran_med_cm=float(np.median(te)))
  print(gw, res[gw], flush=True)
best = max(res, key=lambda g: res[g]["crit"])
json.dump({"weights": {str(g): v for g, v in res.items()}, "chosen": best}, open(os.path.join(EXP, "guide_tune.json"), "w"), indent=1)
print("chosen guide_w", best)
