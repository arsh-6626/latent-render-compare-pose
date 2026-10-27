"""SPEED+ (Tango spacecraft, single object) as a LiePose crop dataset.

Same crop and pose parametrisation as the BOP pipeline (liepose.data.bop.dataset): DZI crop around the object
box, allocentric rotation, translation as (centroid offset / box size, depth / crop focal ratio), standardised by
tran_scale / tran_offset. SPEED+ has no masks, so `mask` is all ones and there is no background augmentation.

Pose convention (checked by projecting tangoPoints.mat onto the images): q_vbs2tango_true is a scalar-first
quaternion whose standard rotation matrix maps Tango-frame points into the camera frame, x_cam = R x + r.
Boxes come from projecting the 11 Tango keypoints with the lens distortion (GT box); a detector box can be
passed instead through `boxes`.

Data root: $SPEEDPLUS_ROOT, default $DATADIR/speedplusv2 (camera.json, tangoPoints.mat, synthetic/, lightbox/, sunlamp/).
"""

import json
import os

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from liepose.data.bop.data_utils.augment import Augment, build_bbox_dzi_augment, build_color_augment
from liepose.data.bop.data_utils.transform import CropResizeAffiner
from liepose.data.bop.dataset import collate_fn
from liepose.data.utils import repeat_dataloader, seed_worker

SPLITS = {
  "train": ("synthetic", "train.json"),
  "val": ("synthetic", "validation.json"),
  "val_aug": ("synthetic", "validation.json"),  # val crops with the training colour augmentation (scripts/speedplus/augment_crops.py)
  "lightbox": ("lightbox", "test.json"),
  "sunlamp": ("sunlamp", "test.json"),
}
OBJECT_ID = 1


def speedplus_root():
  return os.environ.get("SPEEDPLUS_ROOT", os.path.join(os.environ.get("DATADIR", "./"), "speedplusv2"))


def load_camera(root=None):
  root = root or speedplus_root()
  cam = json.load(open(os.path.join(root, "camera.json")))
  return np.asarray(cam["cameraMatrix"], np.float64), np.asarray(cam["distCoeffs"], np.float64)


def load_keypoints(root=None):
  root = root or speedplus_root()
  return sio.loadmat(os.path.join(root, "tangoPoints.mat"))["tango3Dpoints"].T.astype(np.float64)  # (11,3)


def quat_wxyz_to_mat(q):
  w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def load_split(split, root=None):
  """Annotations of one split: file paths, quat (wxyz), tran (m), GT box (xyxy, clipped), centroid_2d."""
  root = root or speedplus_root()
  sub, fname = SPLITS[split]
  entries = json.load(open(os.path.join(root, sub, fname)))
  K, D = load_camera(root)
  kpts = np.concatenate([load_keypoints(root), np.zeros((1, 3))])  # last row: object origin
  n = len(entries)
  out = {"file": [], "quat": np.zeros((n, 4), np.float32), "tran": np.zeros((n, 3), np.float32),
         "bbox": np.zeros((n, 4), np.float32), "centroid_2d": np.zeros((n, 2), np.float32)}
  for i, e in enumerate(entries):
    q = np.asarray(e["q_vbs2tango_true"], np.float64)
    t = np.asarray(e["r_Vo2To_vbs_true"], np.float64)
    uv, _ = cv2.projectPoints(kpts, cv2.Rodrigues(quat_wxyz_to_mat(q))[0], t, K, D)
    uv = uv[:, 0]
    x1, y1 = uv[:-1].min(0)
    x2, y2 = uv[:-1].max(0)
    out["file"].append(os.path.join(root, sub, "images", e["filename"]))
    out["quat"][i] = q / np.linalg.norm(q)
    out["tran"][i] = t
    out["bbox"][i] = [max(x1, 0), max(y1, 0), min(x2, 1920), min(y2, 1200)]
    out["centroid_2d"][i] = uv[-1]
  return out, K.astype(np.float32)


class SpeedPlusPoseDataset(Dataset):
  def __init__(self, split, input_res=224, is_train=True, augment={}, tran_scale=(1.0, 1.0, 1.0),
               tran_offset=(0.0, 0.0, 0.0), boxes=None, root=None, **kwargs):
    self.split = split
    self.ann, self.cam = load_split(split, root)
    if boxes is not None:  # detector boxes (N,4) xyxy, same order as the split json
      self.ann["bbox"] = np.asarray(boxes, np.float32)
    self.height, self.width = 1200, 1920
    self.input_res = input_res
    self.is_train = is_train
    self.augment = Augment(**augment)
    self.tran_scale = np.asarray(tran_scale, np.float32)
    self.tran_offset = np.asarray(tran_offset, np.float32)
    if is_train:
      self.color_augment = build_color_augment()
      self.bbox_dzi_augment = build_bbox_dzi_augment(self.augment.dzi_pad_scale, self.augment.dzi_scale_ratio,
                                                     self.augment.dzi_shift_ratio, self.augment.dzi_type)

  def __len__(self):
    return len(self.ann["file"])

  def __getitem__(self, index):
    a = self.ann
    image = cv2.imread(a["file"][index], cv2.IMREAD_GRAYSCALE)

    bbox_xyxy = a["bbox"][index]
    x1, y1, x2, y2 = bbox_xyxy
    bbox_w, bbox_h = max(x2 - x1, 1), max(y2 - y1, 1)
    if self.is_train and self.augment.random_bbox_dzi:
      bbox_center, dzi_size = self.bbox_dzi_augment(bbox_xyxy, self.height, self.width)
    else:
      bbox_center = 0.5 * np.asarray([x1 + x2, y1 + y2], np.float32)
      dzi_size = max(bbox_w, bbox_h) * self.augment.dzi_pad_scale
    dzi_size = min(dzi_size, max(self.height, self.width)) * 1.0

    F = CropResizeAffiner(self.cam, bbox_center, dzi_size, self.input_res)
    roi_image = F.transform_image(image)  # (res,res) uint8
    roi_image = np.repeat(roi_image[..., None], 3, axis=-1)
    # colour augmentation on the crop (much cheaper than on the 1920x1200 frame)
    if self.is_train and np.random.rand() < self.augment.random_color_prob:
      roi_image = self.color_augment.augment_image(roi_image)
      roi_image = np.repeat(roi_image.mean(-1, keepdims=True), 3, axis=-1)  # SPEED+ is grayscale
    roi_image = roi_image.astype(np.float32)
    quat, tran = a["quat"][index], a["tran"][index]
    quat_allo, tran_allo = F.transform_pose(quat, tran)
    focal_ratio = F.focal_ratio[0]
    delta_c = a["centroid_2d"][index] - bbox_center
    tran_ratio = np.asarray([bbox_w, bbox_h, focal_ratio], np.float32) * self.tran_scale
    tran_scaled = np.array([delta_c[0], delta_c[1], tran_allo[2]], np.float32) / tran_ratio - self.tran_offset

    return {
      "index": index,
      "object_id": OBJECT_ID,
      "image": roi_image,
      "mask": np.ones((self.input_res, self.input_res, 1), np.float32),
      "bbox": np.asarray(bbox_xyxy, np.float32),
      "bbox_center": np.asarray(bbox_center, np.float32),
      "dzi_size": np.float32(dzi_size),
      "focal_ratio": np.float32(focal_ratio),
      "cam": self.cam,
      "rotation": quat,  # wxyz
      "rotation_allo": quat_allo,  # wxyz
      "translation": tran,
      "translation_allo": tran_allo,
      "translation_scaled": tran_scaled,
      "translation_ratio": tran_ratio,
      "translation_offset": self.tran_offset,
    }

  def get_batch(self, indices):
    return collate_fn([self[int(i)] for i in np.asarray(indices).flatten()])


def recover_translation(tran_scaled, bbox, bbox_center, focal_ratio, tran_scale, tran_offset, root=None):
  """Invert the parametrisation: scaled translation (N,3) -> camera-frame translation in metres (N,3).
  The centroid offset is in distorted pixels (as in the forward direction), so it is undistorted here."""
  K, D = load_camera(root)
  bw = np.maximum(bbox[:, 2] - bbox[:, 0], 1)
  bh = np.maximum(bbox[:, 3] - bbox[:, 1], 1)
  ratio = np.stack([bw, bh, focal_ratio], -1) * np.asarray(tran_scale)
  d = (np.asarray(tran_scaled) + np.asarray(tran_offset)) * ratio
  c = (bbox_center + d[:, :2]).astype(np.float64).reshape(-1, 1, 2)
  xy = cv2.undistortPoints(c, K, D)[:, 0]  # normalised image coordinates
  z = d[:, 2]
  return np.stack([xy[:, 0] * z, xy[:, 1] * z, z], -1)


class TestDataset:
  def __init__(self, *args, batch_size, **kwargs):
    self.batch_size = batch_size
    self.dataset = SpeedPlusPoseDataset(*args, is_train=False, **kwargs)
    num = len(self.dataset)
    self.batches = np.array_split(np.arange(num), np.arange(num, step=batch_size))[1:]

  def __iter__(self):
    self.iter = iter(self.batches)
    return self

  def __next__(self):
    return self.dataset.get_batch(next(self.iter))

  def __len__(self):
    return len(self.batches)


class TrainDataset:
  def __init__(self, *args, batch_size, seed, num_workers, **kwargs):
    self.batch_size = batch_size
    self.seed = seed
    self.num_workers = num_workers
    self.dataset = SpeedPlusPoseDataset(*args, is_train=True, **kwargs)

  def __iter__(self):
    generator = torch.Generator()
    generator.manual_seed(self.seed)
    return repeat_dataloader(DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                        num_workers=self.num_workers, worker_init_fn=seed_worker, drop_last=True,
                                        generator=generator, collate_fn=collate_fn))

  def __len__(self):
    return None
