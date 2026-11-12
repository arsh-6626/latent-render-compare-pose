"""Mathematics used by the scenes (numpy only). Verified against jaxlie and the paper's code in tools/verify_math.py.

Conventions (as LiePose / jaxlie): an se(3) tangent vector is [v (3, translation part), w (3, rotation part)];
Exp is the coupled SE(3) exponential, translation = V(w) v. A pose is (R, t) as a 3x3 matrix and a 3-vector.
"""
import numpy as np


def hat(w):
  return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=float)


def so3_exp(w):
  th = np.linalg.norm(w)
  K = hat(w)
  if th < 1e-8:
    return np.eye(3) + K + 0.5 * K @ K
  return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def so3_log(R):
  c = np.clip((np.trace(R) - 1) / 2, -1, 1)
  th = np.arccos(c)
  if th < 1e-7:
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
  if np.pi - th < 1e-5:  # near pi: axis from the symmetric part
    A = (R + np.eye(3)) / 2
    i = np.argmax(np.diag(A))
    ax = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
    return ax * th
  return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def V_mat(w):
  th = np.linalg.norm(w)
  K = hat(w)
  if th < 1e-8:
    return np.eye(3) + 0.5 * K
  return np.eye(3) + (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K


def se3_exp(xi):
  """xi = [v, w] -> (R, t)"""
  v, w = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
  return so3_exp(w), V_mat(w) @ v


def se3_log(R, t):
  w = so3_log(R)
  return np.concatenate([np.linalg.solve(V_mat(w), t), w])


def compose(g, h):
  (R1, t1), (R2, t2) = g, h
  return R1 @ R2, R1 @ t2 + t1


def inverse(g):
  R, t = g
  return R.T, -R.T @ t


def sigmas(T=100, start=1e-8, end=1.0, power=3.0):
  """LiePose's VE power-law schedule: variance a_t = linspace(start^(1/p), end^(1/p), T+1)^p, sigma_t = sqrt(a_t)"""
  a = np.linspace(start ** (1 / power), end ** (1 / power), T + 1) ** power
  return np.sqrt(a)


def forward(g0, sigma, eps):
  """g_t = g0 Exp(sigma eps)"""
  return compose(g0, se3_exp(sigma * np.asarray(eps)))


def geodesic(g0, g1, tau):
  """flow-matching path g_tau = g0 Exp(tau xi), xi = Log(g0^-1 g1)"""
  xi = se3_log(*compose(inverse(g0), g1))
  return compose(g0, se3_exp(tau * xi)), xi


def uniform_rotations(n, rng):
  """Haar-uniform rotations (unit quaternions from a normalised Gaussian)"""
  q = rng.normal(size=(n, 4))
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  w, x, y, z = q.T
  return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                   np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                   np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def volume_weights(sigma, delta):
  """standard volume rendering along a ray: alpha_j = 1 - exp(-sigma_j delta), T_j = prod_{i<j}(1 - alpha_i), w_j = T_j alpha_j
  (as TlessField.__call__)"""
  alpha = 1 - np.exp(-sigma * delta)
  T = np.cumprod(np.concatenate([[1.0], 1 - alpha[:-1] + 1e-10]))
  return alpha, T, T * alpha


def energy(z_hat, z, w):
  """E(g; z) = sum_p w_p (1/D) ||z_hat_p - z_p||^2 / sum_p w_p"""
  return float((w * ((z_hat - z) ** 2).mean(-1)).sum() / w.sum())
