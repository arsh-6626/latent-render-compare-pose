"""Geometry tokens (probe_ret.geometry) for SPEED+ crops: real camera, LiePose's crop-relative pose.

A pose is LiePose's (allocentric R, translation_scaled) relative to its crop; with the crop's box it maps to the
camera frame exactly as SpeedPlusPoseDataset / recover_translation define it, except that lens distortion is
ignored on both sides (pinhole K throughout; geometry is a model input, not a metric). Per-crop meta
k = (s, x0, y0, bw, bh, focal_ratio): crop pixel = s (u - (x0, y0)), s = 224 / dzi_size, (x0, y0) = bbox_center -
dzi_size / 2, and (bw, bh) the box size used by the translation parametrisation. These come from the box only (the
observation's crop); no pose enters them.
"""

import glob
import os

import jax.numpy as jnp
import numpy as np

from liepose.data.speedplus.dataset import load_camera, load_keypoints
from liepose.exp.speedplus.run import TRAN_OFFSET, TRAN_SCALE
from liepose.utils.jnpfn import get_allo_rotation

META_KEYS = ["index", "translation_scaled", "bbox", "bbox_center", "dzi_size", "focal_ratio"]


def crop_meta(crops_dir):
  """(N, 6) meta from export_crops shards (box quantities only) + index / translation_scaled for alignment checks"""
  sh = [np.load(f) for f in sorted(glob.glob(os.path.join(crops_dir, "shard_*.npz")))]
  d = {k: np.concatenate([s[k] for s in sh]) for k in META_KEYS}
  return crop_k(d).astype(np.float32), d["index"], d["translation_scaled"]


def crop_k(d):
  """(N, 6) crop meta (s, x0, y0, bw, bh, focal_ratio) from bbox, bbox_center, dzi_size, focal_ratio"""
  dzi = d["dzi_size"].astype(np.float64)
  bb = d["bbox"].astype(np.float64)
  return np.stack([224.0 / dzi, d["bbox_center"][:, 0] - dzi / 2, d["bbox_center"][:, 1] - dzi / 2,
                   np.maximum(bb[:, 2] - bb[:, 0], 1), np.maximum(bb[:, 3] - bb[:, 1], 1), d["focal_ratio"]], -1)


def quat_to_mat(q):
  """(..., 4) wxyz -> (..., 3, 3)"""
  q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def crop_feat(k):
  """crop meta (s, x0, y0, bw, bh, focal_ratio) -> log zoom, crop centre in normalised image coordinates, log box
  aspect (the denoiser's crop conditioning; the box only, no pose)"""
  s = k[..., :1]
  ctr = (k[..., 1:3] + 112.0 / s - jnp.array([960.0, 600.0])) / jnp.array([960.0, 600.0])
  return jnp.concatenate([jnp.log(s), ctr, jnp.log(k[..., 3:4] / k[..., 4:5])], -1)


class SpeedPlusGeometry:
  """callable with probe_ret.geometry's signature: (Rr, tr, Rg, tg, grid, kr, kg) -> (tok, vpt)"""

  def __init__(self, root=None):
    K, _ = load_camera(root)
    self.f = jnp.array([K[0, 0], K[1, 1]], jnp.float32)
    self.c = jnp.array([K[0, 2], K[1, 2]], jnp.float32)
    self.obj_r = float(np.linalg.norm(load_keypoints(root), axis=-1).max())
    self.scale = jnp.array(TRAN_SCALE, jnp.float32)
    self.offset = jnp.array(TRAN_OFFSET, jnp.float32)

  def __hash__(self):  # flax module attribute
    return id(self)

  def cam_pose(self, R, ts, k):
    """(allocentric R, translation_scaled) in crop k -> (egocentric R, camera-frame t in metres)"""
    d = (ts + self.offset) * jnp.concatenate([k[..., 3:5], k[..., 5:6]], -1) * self.scale
    c = k[..., 1:3] + 112.0 / k[..., :1] + d[..., :2]  # bbox_center + centroid offset, image pixels
    z = jnp.maximum(d[..., 2:3], 0.5)  # noisy denoiser hypotheses can put the object behind the camera
    t = jnp.concatenate([(c - self.c) / self.f * z, z], -1)
    A = quat_to_mat(get_allo_rotation(t))  # ego -> allo, CropResizeAffiner.transform_pose
    return jnp.swapaxes(A, -1, -2) @ R, t

  def project(self, P):
    return self.f * P[..., :2] / jnp.maximum(P[..., 2:3], 0.05) + self.c

  @staticmethod
  def to_crop(u, k):
    return k[..., None, :1] * (u - k[..., None, 1:3])

  @staticmethod
  def from_crop(u, k):
    return u / k[..., None, :1] + k[..., None, 1:3]

  def __call__(self, Rr, tr, Rg, tg, grid, kr, kg, logd=None):
    """logd (..., P): per-token log depth relative to the object centre's depth (world model v4: exact projection of
    each reference token at its own depth); None: every token on the plane z = Zr (v1-v3)"""
    Rr, tr = self.cam_pose(Rr, tr, kr)
    Rg, tg = self.cam_pose(Rg, tg, kg)
    patch = 224.0 / grid
    cc = (jnp.arange(grid) + 0.5) * patch
    pc = jnp.stack(jnp.meshgrid(cc, cc, indexing="xy"), -1).reshape(-1, 2)
    p = self.from_crop(jnp.broadcast_to(pc, tr.shape[:-1] + pc.shape), kr)  # reference token centres, image px
    Zr = tr[..., 2][..., None, None]
    Pc = jnp.concatenate([(p - self.c) / self.f * Zr, jnp.broadcast_to(Zr, p.shape[:-1] + (1,))], -1)  # plane z = Zr
    if logd is not None:
      Pc = Pc * jnp.exp(logd)[..., None]  # along each token's ray
    o = jnp.einsum("...ji,...kj->...ki", Rr, Pc - tr[..., None, :])  # object frame
    Pg = jnp.einsum("...ij,...kj->...ki", Rg, o) + tg[..., None, :]
    tok = jnp.concatenate([jnp.broadcast_to(pc / 112.0 - 1.0, Pg.shape[:-1] + (2,)),
                           (self.to_crop(self.project(Pg), kg) - pc) / patch, jnp.log(jnp.maximum(Pg[..., 2:3], 0.05) / Zr)], -1)
    V = self.obj_r * jnp.concatenate([jnp.zeros((1, 3)), jnp.eye(3), -jnp.eye(3)])
    Vr = jnp.einsum("...ij,kj->...ki", Rr, V) + tr[..., None, :]
    Vg = jnp.einsum("...ij,kj->...ki", Rg, V) + tg[..., None, :]
    ur, ug = self.to_crop(self.project(Vr), kr), self.to_crop(self.project(Vg), kg)
    vpt = jnp.concatenate([ur / 112.0 - 1.0, ug / 112.0 - 1.0, (ug - ur) / patch, Vr[..., 2:] / Zr - 1.0,
                           Vg[..., 2:] / Zr - 1.0], -1)
    return tok, vpt
