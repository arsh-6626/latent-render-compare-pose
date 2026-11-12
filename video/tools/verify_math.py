"""Numerical checks of video/scenes/larc_math.py against jaxlie and the paper's code / outputs (liepose env, CPU).
  JAX_PLATFORMS=cpu PYTHONPATH=src:video/scenes /home/kavinder/miniconda3/envs/liepose/bin/python video/tools/verify_math.py
"""
import numpy as np
import jax.numpy as jnp
from jaxlie import SE3, SO3

import larc_math as lm
from liepose.noise.ve import PowerNoiseSchedule

rng = np.random.default_rng(0)
D = "/media/kavinder/hdd3/larc_video/data"
ok = True


def check(name, err, tol):
  global ok
  print(f"{name:55s} max err {err:.2e}  {'OK' if err < tol else 'FAIL'}")
  ok &= err < tol


# SE(3) Exp / Log vs jaxlie (tangent order [v, w])
e1 = e2 = e3 = 0
for _ in range(200):
  xi = rng.normal(size=6) * rng.uniform(0.1, 1.5)
  R, t = lm.se3_exp(xi)
  G = SE3.exp(jnp.asarray(xi))
  e1 = max(e1, np.abs(R - np.asarray(G.rotation().as_matrix())).max(), np.abs(t - np.asarray(G.translation())).max())
  if np.linalg.norm(xi[3:]) < 3.0:
    e2 = max(e2, np.abs(lm.se3_log(R, t) - xi).max())
  g0 = lm.se3_exp(rng.normal(size=6))
  g1 = lm.se3_exp(rng.normal(size=6))
  gc = lm.compose(g0, g1)
  Gc = SE3.exp(jnp.asarray(lm.se3_log(*g0))) @ SE3.exp(jnp.asarray(lm.se3_log(*g1)))
  e3 = max(e3, np.abs(gc[0] - np.asarray(Gc.rotation().as_matrix())).max())
check("se3_exp vs jaxlie SE3.exp", e1, 1e-5)
check("se3_log(se3_exp(xi)) = xi (|w| < 3)", e2, 1e-5)
check("composition vs jaxlie", e3, 1e-5)

# schedule vs LiePose's PowerNoiseSchedule(1e-8, 1, 100, 3).sqrt_alphas
ns = PowerNoiseSchedule(1e-8, 1.0, 100, 3)
check("sigma_t vs PowerNoiseSchedule.sqrt_alphas", np.abs(lm.sigmas() - np.asarray(ns.sqrt_alphas)).max(), 1e-6)
print("  sigma_0, sigma_50, sigma_100 =", lm.sigmas()[[0, 50, 100]])

# forward process g_t = g0 Exp(sigma eps) vs jaxlie
g0 = lm.se3_exp(rng.normal(size=6))
eps = rng.normal(size=6)
gt = lm.forward(g0, 0.7, eps)
Gt = SE3.exp(jnp.asarray(lm.se3_log(*g0))) @ SE3.exp(jnp.asarray(0.7 * eps))
check("forward process vs jaxlie", np.abs(gt[1] - np.asarray(Gt.translation())).max(), 1e-5)

# flow-matching geodesic endpoints
g1 = lm.se3_exp(rng.normal(size=6))
ga, _ = lm.geodesic(g0, g1, 0.0)
gb, _ = lm.geodesic(g0, g1, 1.0)
check("geodesic(0) = g0, geodesic(1) = g1", max(np.abs(ga[0] - g0[0]).max(), np.abs(gb[1] - g1[1]).max()), 1e-6)

# real trajectories: tan repr -> pose with our Exp equals the saved poses
TR = np.load(f"{D}/traj.npz")
rt, Rs, Ts = TR["field_full_lat_rt"], TR["field_full_lat_R"], TR["field_full_lat_T"]
e = max(max(np.abs(lm.se3_exp(rt[i, j])[0] - Rs[i, j]).max(), np.abs(lm.se3_exp(rt[i, j])[1] - Ts[i, j]).max())
        for i in range(0, 101, 10) for j in range(16))
check("trajectory tan repr -> (R, t) with se3_exp", e, 1e-4)

# volume rendering weights: per-ray weights recomputed from sigma, and 2x2-pooled opacity vs the field's own output
RY = np.load(f"{D}/rays.npz")
TK = np.load(f"{D}/tokens.npz")
sig, w_saved, dd = RY["gt_sigma"], RY["gt_w"], RY["gt_d"]
e = 0
for i in range(256):
  delta = 2 * float(RY["r_o"]) / 32 * np.linalg.norm(dd[i])
  _, _, w = lm.volume_weights(sig[i], delta)
  e = max(e, np.abs(w - w_saved[i]).max())
check("volume weights from sigma (per ray)", e, 1e-5)
op16 = RY["gt_opa16"].reshape(8, 2, 8, 2).mean((1, 3)).reshape(64)
check("pooled opacity = field output opacity", np.abs(op16 - TK["op_gt"]).max(), 1e-4)
check("sum of weights in [0, 1]", max(0, w_saved.sum(-1).max() - 1), 1e-6)

# energy formula vs the field run's energies
check("energy(z_hat_gt, z, w) vs E_gt", abs(lm.energy(TK["zh_gt"], TK["z"], TK["w"]) - float(TK["E_gt"])), 1e-4)
check("energy(z_hat_wrong, z, w) vs E_wrong", abs(lm.energy(TK["zh_wr"], TK["z"], TK["w"]) - float(TK["E_wr"])), 1e-4)
L = np.load(f"{D}/landscape.npz")
check("landscape centre = E at GT", abs(L["E"][18, 36] - float(TK["E_gt"])), 1e-4)

# uniform rotations: mean of cos(angle) of Haar rotations is -1/2 ... check angle density ~ (1 - cos th)/pi
Rr = lm.uniform_rotations(200000, rng)
ang = np.arccos(np.clip((np.trace(Rr, axis1=1, axis2=2) - 1) / 2, -1, 1))
h, b = np.histogram(ang, bins=20, range=(0, np.pi), density=True)
c = (b[1:] + b[:-1]) / 2
check("Haar angle density (1 - cos)/pi", np.abs(h - (1 - np.cos(c)) / np.pi).max(), 0.02)
print("ALL OK" if ok else "SOME CHECKS FAILED")
