"""Probes of the trained LaRC (joint, T-LESS, 100k x 32) on test crops; no training, nothing tuned.

  align    along LiePose's 100-step sampler (300 test crops, one chain each), the cosine between the denoiser's predicted
           noise eps_theta(g_t) and the energy gradient grad_xi E(g_t Exp(xi)) of the jointly trained field, per noise level,
           for the full 6-vector and its rotation and translation parts (the residual-input motivation predicts alignment
           at low noise)
  link     per test crop: the field's prediction error at the GT pose (joint space FVU, wm_diag) and how often its energy
           ranks the GT above 10/20-degree perturbations, against LaRC's own error (first of its 16 test samples)
  runtime  wall-clock per target of the encoder, one field render and one denoiser call (batch 100, RTX 6000 Ada),
           for LaRC and the control; LiePose from its own test log on the same GPU
Run: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false BOP_DS=tless BOP_RENDERER=vispy DATADIR=... CACHE_ROOT_PATH=...
     PYTHONPATH=LARp/src:LARp/lib/bop_toolkit python lrc_probe.py
"""
import csv
import glob
import json
import os
import sys
import time
import types

for _m in ("bop_toolkit_lib.renderer", "bop_toolkit_lib.pose_error"):
  sys.modules.setdefault(_m, types.ModuleType(_m))
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from liepose.exp.cmp import tless_field_dit as tfd
from liepose.exp.cmp.sp_field_dit import FtFieldDenoiser
from liepose.exp.cmp.tless_dit import mask_fg
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.tless_field import TlessGeometry
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.utils.config import get_args, load_config_from_args
from liepose.utils.exploration import Options

DD = os.environ["DATADIR"]; OUT = "/media/kavinder/hdd3/larc_figs/data"; W = "/media/kavinder/hdd3/larc_figs/probe"
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


def testbed(den_mode, joint, ckpt):
  name = f"train_steps=100000+seed=42+den_mode={den_mode}+field=tl_field_v1" + ("+joint=0.999" if joint else "")
  exp = os.path.join(W, name); os.makedirs(os.path.join(exp, "ckpt"), exist_ok=True)
  link = os.path.join(exp, "ckpt", os.path.basename(ckpt))
  if not os.path.exists(link):
    os.symlink(ckpt, link)
  sys.argv = [sys.argv[0], "--root_path", W, f"den_mode=[{den_mode}]", "field_dir=[/media/kavinder/hdd3/larp_abl/probe/tl_field_v1]",
              "train_steps=[100000]", "seed=[42]", "batch_size=[32]", "n_slices=[16]", "skip_train=[True]", f"joint=[{joint}]"]
  a = get_args(exp_name="tless-field", desc="probe")
  opt = next(iter(load_config_from_args(a, tfd.build_explorations()).expand()))
  opt.pop("sample_options", None)
  tb = tfd.Testbed(opt, exp, setup=True)
  tb.opt = Options(tfd.SAMPLE)
  return tb


tb = testbed("field_full_lat_ft", True, "/media/kavinder/hdd3/larp_abl/a1_ckpt/checkpoint_100001")
model, poses, lm, Lie = tb.model, tb._poses_fn(), tb.lie_metrics, tb.Lie
params = {"params": tb.state.params_ema}
params["params"], fp = tb._split_field(params["params"])
tb.field_params = fp
p_fn = jax.vmap(tb.create_p_sample_fn())
T = tb.noise_schedule.timesteps
cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist(); seq = list(zip(cur, cur[1:] + [0]))
encode = jax.jit(lambda im: model.apply(params, im, method=FtFieldDenoiser.encode))
rep = tb.a.repr_type

tm = np.load(f"{DD}/tless_tokens/test_meta.npz"); N = len(tm["object_id"])
sh = np.load(f"{DD}/tless_crops/test/shard_0000.npz")
kmeta = np.load(f"{DD}/tless_tokens/test_crop_meta.npy").astype(np.float32)
obj_all = tm["object_id"].astype(np.int32); R_all = q2m(tm["rotation_allo"]); t_all = tm["translation_scaled"].astype(np.float32)
rng = np.random.default_rng(21)
res = {}


@jax.jit
def step_probe(key, rt, t, tp, obj, z_obs, w, kg, z_ext):
  B = rt.shape[0]
  key, k = jax.random.split(key)
  Rg, tg = poses(rt)
  z_hat, opa, h = tb._render(obj, Rg, tg, kg, z_obs)
  mu = model.apply(params, rt, jnp.full((B, 1), t, jnp.float32), kg, z_obs, w, z_hat, opa, h, None, z_ext)
  eps = jax.vmap(lambda m: lm.as_lie(m).log())(mu)

  def e_sum(xi):
    rp = jax.vmap(lambda r, x: lm.as_repr(lm.as_lie(r) @ Lie.exp(x), rep))(rt, xi)
    R2, t2 = poses(rp)
    return weighted_energy(tb.field_model.apply(fp, obj, R2, t2, kg, app_stats(z_obs)), z_obs, w).sum()

  gE = jax.grad(e_sum)(jnp.zeros((B, 6), jnp.float32))
  _, rt2 = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
  return key, rt2, eps, gE


def cos(a, b):
  return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


# ------------------------------------------------------------------ align
rows = np.sort(rng.choice(N, 300, replace=False))
prior = jax.jit(jax.vmap(lambda kk: lm.as_repr(Lie.sample_uniform(kk), rep)))
C = {"all": [], "rot": [], "tran": []}
for s in range(0, len(rows), 100):
  r = rows[s:s + 100]
  img = jnp.asarray(sh["image"][r].astype(np.float32) * (sh["mask"][r][..., None] > 0))
  w = FG_FLOOR + mask_fg(jnp.asarray((sh["mask"][r] > 0).astype(np.float32)))
  z_obs, z_ext = encode(img)
  key = jax.random.PRNGKey(3 + s); key, k1 = jax.random.split(key)
  rt = prior(jax.random.split(k1, len(r)))
  per = {"all": [], "rot": [], "tran": []}
  for tt, tp in seq:
    key, rt, eps, gE = step_probe(key, rt, jnp.int32(tt), jnp.int32(tp), jnp.asarray(obj_all[r]), z_obs, w, jnp.asarray(kmeta[r]), z_ext)
    eps, gE = np.asarray(eps), np.asarray(gE)
    tr = slice(0, 3) if tb.Lie.__name__ == "SE3" else slice(0, 3)  # jaxlie SE3 tangent: [translation, rotation]
    per["all"].append(cos(eps, gE)); per["tran"].append(cos(eps[:, :3], gE[:, :3])); per["rot"].append(cos(eps[:, 3:], gE[:, 3:]))
  for kk in C:
    C[kk].append(np.stack(per[kk]))  # (100 steps, crops)
sig = np.asarray(tb.noise_schedule.sqrt_alphas)[np.array([c for c, _ in seq])]
out = {}
for kk in C:
  M = np.concatenate(C[kk], 1)
  out[kk] = {f"step {i + 1} (sigma {sig[i]:.3g})": round(float(np.median(M[i])), 3) for i in (0, 24, 49, 74, 89, 94, 99)}
res["align_median_cos(eps, gradE)"] = out
print("align", json.dumps(out, indent=1), flush=True)

# ------------------------------------------------------------------ link
smp = np.load(f"{OUT}/a1_sel16_samples.npz")
row_of = {int(i): j for j, i in enumerate(tm["index"])}
srow = np.array([row_of[int(i)] for i in smp["index"]])
pose0 = smp["pose"][:, 0]  # (wxyz, xyz) of the first sample
geo = TlessGeometry()
re = np.array([rot_err(R_all[srow[j]], q2m(pose0[j, :4][None])[0], obj_all[srow[j]]) for j in range(len(srow))])
d1 = np.load(f"{OUT}/wm_diag_exp1.npz")
fvu = d1["joint_fvu_field"][srow]
# ranking per crop with the joint field (EMA-encoder tokens)
sub = np.sort(rng.choice(len(srow), 1500, replace=False)); r = srow[sub]
rank = np.zeros(len(r))
fe = jax.jit(lambda o, R1, t1, k1, z1, w1: weighted_energy(tb.field_model.apply(fp, o, R1, t1, k1, app_stats(z1)), z1, w1))
for s in range(0, len(r), 100):
  rr = r[s:s + 100]
  img = jnp.asarray(sh["image"][rr].astype(np.float32) * (sh["mask"][rr][..., None] > 0))
  w = FG_FLOOR + mask_fg(jnp.asarray((sh["mask"][rr] > 0).astype(np.float32)))
  z_obs, _ = encode(img)
  Rf = R_all[rr].astype(np.float32)
  eg = np.asarray(fe(jnp.asarray(obj_all[rr]), Rf, t_all[rr], kmeta[rr], z_obs, w))
  wins = []
  for dl in (10, 20):
    for _ in range(8):
      ax = rng.normal(size=(len(rr), 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
      Rp = (Rotation.from_rotvec(ax * np.radians(dl)).as_matrix() @ R_all[rr]).astype(np.float32)
      wins.append(np.asarray(fe(jnp.asarray(obj_all[rr]), Rp, t_all[rr], kmeta[rr], z_obs, w)) > eg)
  rank[s:s + len(rr)] = np.mean(wins, 0)
re_s = re[sub]
out = dict(spearman_rot_err_vs_fvu=list(map(float, spearmanr(fvu, re))),
           spearman_rot_err_vs_ranking=list(map(float, spearmanr(rank, re_s))),
           correct_rate_by_fvu_quartile=[float((re[(fvu >= a) & (fvu < b)] < 10).mean()) for a, b in
                                         zip(np.percentile(fvu, [0, 25, 50, 75]), list(np.percentile(fvu, [25, 50, 75])) + [np.inf])],
           correct_rate_by_ranking_bin={f"{a}-{b}": float((re_s[(rank >= a) & (rank < b)] < 10).mean()) if ((rank >= a) & (rank < b)).sum() else None
                                        for a, b in ((0, 0.75), (0.75, 0.95), (0.95, 1.01))},
           ranking_bin_counts=[int(((rank >= a) & (rank < b)).sum()) for a, b in ((0, 0.75), (0.75, 0.95), (0.95, 1.01))])
res["link"] = out
print("link", json.dumps(out, indent=1), flush=True)

# ------------------------------------------------------------------ runtime
B = 100; r = np.arange(B)
img = jnp.asarray(sh["image"][r].astype(np.float32)); w = FG_FLOOR + mask_fg(jnp.asarray((sh["mask"][r] > 0).astype(np.float32)))
rt = prior(jax.random.split(jax.random.PRNGKey(0), B)); kg = jnp.asarray(kmeta[r]); ob = jnp.asarray(obj_all[r])
z_obs, z_ext = encode(img)
ren = jax.jit(lambda rt_: tb._render(ob, *poses(rt_), kg, z_obs))
z_hat, opa, h = ren(rt)
den = jax.jit(lambda rt_, zh, op, hh: model.apply(params, rt_, jnp.full((B, 1), 50, jnp.float32), kg, z_obs, w, zh, op, hh, None, z_ext))
den(rt, z_hat, opa, h)


def timeit(f, n=20):
  jax.block_until_ready(f()); t0 = time.time()
  for _ in range(n):
    x = f()
  jax.block_until_ready(x); return (time.time() - t0) / n / B * 1000  # ms per target


t_enc = timeit(lambda: encode(img)); t_ren = timeit(lambda: ren(rt)); t_den = timeit(lambda: den(rt, z_hat, opa, h))
lp = []
for f in glob.glob(f"{DD}/logs/tless_test/liepose_se3_400k_noema/*.csv"):
  lp += [float(x["time"]) for x in csv.DictReader(open(f))]
res["runtime_ms_per_target"] = dict(gpu="RTX 6000 Ada", encoder=t_enc, field_per_step=t_ren, denoiser_per_step=t_den,
                                    larc_total_100_steps=t_enc + 100 * (t_ren + t_den), control_total_100_steps=t_enc + 100 * t_den,
                                    liepose_test_log_time_median_s=float(np.median(lp)) if lp else None,
                                    note="LiePose: the 'time' column of its BOP CSV (seconds, as written by the evaluator)")
print("runtime", json.dumps(res["runtime_ms_per_target"]), flush=True)
json.dump(res, open(f"{OUT}/lrc_probe.json", "w"), indent=1)
print("saved")
