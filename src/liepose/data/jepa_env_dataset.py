import math
from typing import Any, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class JEPAEnvDataset:
  """3D Environment Trajectory & Pose Dataset for D-JEPA World Model + LiePose Joint Learning."""

  def __init__(
      self,
      batch_size: int = 16,
      img_res: int = 224,
      action_dim: int = 6,
      num_samples: int = 1000,
      seed: int = 42,
  ):
    self.batch_size = batch_size
    self.img_res = img_res
    self.action_dim = action_dim
    self.num_samples = num_samples
    self.seed = seed
    self.rng = np.random.RandomState(seed)

  def _generate_synthetic_sample(
      self, idx: int
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic 3D environment step with observation, action, next obs, and pose."""
    # Synthetic observation image with 3D geometric shape (e.g. textured object rendering)
    H = W = self.img_res
    
    # Base pattern
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    dist = np.sqrt(grid_x**2 + grid_y**2)
    
    phase = (idx % 100) * 0.1
    r = np.sin(dist * 10.0 + phase) * 0.5 + 0.5
    g = np.cos(grid_x * 5.0 + phase) * 0.5 + 0.5
    b = np.sin(grid_y * 5.0 + phase) * 0.5 + 0.5
    
    img_obs = np.stack([r, g, b], axis=-1).astype(np.float32)
    
    # 3D Action: relative SE(3) step (translation dx, dy, dz, rotation d_roll, d_pitch, d_yaw)
    action = self.rng.randn(self.action_dim).astype(np.float32) * 0.1
    
    # Next observation image resulting from action step
    img_next = np.roll(img_obs, shift=int(action[0] * 10), axis=0)
    img_next = np.roll(img_next, shift=int(action[1] * 10), axis=1)
    
    # Ground-truth rotation quaternion (wxyz)
    angle = self.rng.uniform(0, 2 * math.pi)
    axis = self.rng.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    sin_a2 = math.sin(angle / 2)
    cos_a2 = math.cos(angle / 2)
    rot_gt = np.array([cos_a2, axis[0] * sin_a2, axis[1] * sin_a2, axis[2] * sin_a2], dtype=np.float32)
    
    # Ground-truth translation vector
    trans_gt = self.rng.randn(3).astype(np.float32) * 0.5

    return img_obs, action, img_next, rot_gt, trans_gt

  def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
    idx = 0
    while True:
      batch_img = []
      batch_action = []
      batch_target_img = []
      batch_rot = []
      batch_trans = []

      for _ in range(self.batch_size):
        img_obs, action, img_next, rot_gt, trans_gt = self._generate_synthetic_sample(idx)
        batch_img.append(img_obs)
        batch_action.append(action)
        batch_target_img.append(img_next)
        batch_rot.append(rot_gt)
        batch_trans.append(trans_gt)
        idx += 1

      yield {
          "image": np.array(batch_img, dtype=np.float32),
          "action": np.array(batch_action, dtype=np.float32),
          "target_image": np.array(batch_target_img, dtype=np.float32),
          "rot_gt": np.array(batch_rot, dtype=np.float32),
          "trans_gt": np.array(batch_trans, dtype=np.float32),
      }
