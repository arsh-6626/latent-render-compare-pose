"""SPEED+ test of a trained LiePose checkpoint: full splits, official ESA metric.

Inference sees only the image crop (cut from the keypoint box, as at training time) and the camera intrinsics.
Sampling is LiePose's own test loop (Testbed.run_test: uniform SE(3) start, 100-step DDIM, beta_skip, 1 sample per
image, final rt as the estimate, as in data/bop/evaluator.py). The GT pose is read only by the scorer.

The estimate (allocentric rotation, scaled translation) goes to the camera frame with recover_translation and
the inverse of CropResizeAffiner's allocentric transform, using the *estimated* translation; it is then scored
with the SPEED+ / SPEC2021 metric (scripts/speedplus/esa_score.py: e_q = 2 arccos|<q,q_gt>|,
e_t = ||t - t_gt|| / ||t_gt||, precision floors 0.169 deg and 2.173 mm/m).

  DATADIR=... PYTHONPATH=src python scripts/speedplus/test_liepose.py \
      --exp "$LOGS/experiments/speedplus-se3/lie_type=se3+repr_type=tan+timesteps=100+seed=42" --step 400000 \
      --splits lightbox sunlamp --out $LOGS/speedplus_test/liepose_se3_400k
"""

import argparse
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esa_score import allo_to_ego, esa  # noqa: E402

from liepose.data.speedplus.dataset import recover_translation  # noqa: E402
from liepose.exp.speedplus.run import TestbedSE3  # noqa: E402
from liepose.utils.exploration import Options  # noqa: E402

# LiePose's test sampling options (liepose.exp.bop.run.build_explorations)
SAMPLE = dict(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=100, n_slices=1)


class Collector:
  """Evaluator stand-in for Testbed.run_test: keeps the model's estimate and the crop geometry only."""

  def __init__(self):
    self.out = {k: [] for k in ["index", "pose", "bbox", "bbox_center", "focal_ratio", "time"]}

  def process(self, inputs, outputs):
    self.out["pose"].append(np.asarray(outputs["seq_rt"][-1]))  # (b, n_slices, 7) wxyz_xyz, final step
    for k in ["index", "bbox", "bbox_center", "focal_ratio"]:
      self.out[k].append(np.asarray(inputs[k]))
    self.out["time"].append(outputs["time"])

  def summarize(self):
    pass


class SpeedPlusTest(TestbedSE3):
  def create_test_evaluator(self, dataset):
    self.collector = Collector()
    return self.collector


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--exp", required=True)
  p.add_argument("--step", type=int, required=True)
  p.add_argument("--splits", nargs="+", default=["lightbox", "sunlamp"])
  p.add_argument("--out", required=True)
  p.add_argument("--n_slices", type=int, default=1, help="samples per image; sample 0 is scored, all are saved")
  a = p.parse_args()
  SAMPLE["n_slices"] = a.n_slices
  os.makedirs(a.out, exist_ok=True)

  cfg = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(a.exp, "configs.yaml")), resolve=True))
  cfg.restore_path, cfg.restore_step, cfg.skip_train = None, a.step, True
  tb = SpeedPlusTest(cfg, a.exp, setup=True)
  assert int(tb.state.step) == a.step, (int(tb.state.step), a.step)
  print(f"restored step {int(tb.state.step)}", flush=True)

  res = {"exp": a.exp, "step": a.step, "sample": SAMPLE}
  for split in a.splits:
    tb.a.test_datasets = split
    tb.run_test(Options(SAMPLE), os.path.join(a.out, split))
    o = {k: np.concatenate(v) if k != "time" else np.asarray(v) for k, v in tb.collector.out.items()}
    ann = tb.create_test_dataset().dataset.ann  # GT, for scoring only
    idx = o["index"].astype(np.int64)
    assert len(idx) == len(ann["quat"]) and (np.sort(idx) == np.arange(len(idx))).all(), "not the full split"
    S = o["pose"].shape[1]
    quat_allo, t_scaled = o["pose"][..., :4], o["pose"][..., 4:].astype(np.float64)
    rep = lambda x: np.repeat(x, S, 0)
    t_est = recover_translation(t_scaled.reshape(-1, 3), rep(o["bbox"]), rep(o["bbox_center"]), rep(o["focal_ratio"]),
                                cfg.tran_scale, cfg.tran_offset)
    R_allo = Rotation.from_quat(np.roll(quat_allo.reshape(-1, 4), -1, -1)).as_matrix()
    R_est, t_est = allo_to_ego(R_allo, t_est).reshape(-1, S, 3, 3), t_est.reshape(-1, S, 3)
    if S == 1:
      quat_allo, t_scaled = quat_allo[:, 0], t_scaled[:, 0]
    else:  # all samples, (n, S, ...)
      np.savez(os.path.join(a.out, f"{split}_samples.npz"), index=idx, R=R_est, t=t_est, quat_allo=quat_allo, t_scaled=t_scaled)
    R_est, t_est = R_est[:, 0], t_est[:, 0]
    R_gt = Rotation.from_quat(np.roll(ann["quat"][idx], -1, -1)).as_matrix()
    t_gt = ann["tran"][idx].astype(np.float64)
    r = esa(R_est, t_est, R_gt, t_gt)
    r.update(n=int(len(idx)), sec_per_batch=float(o["time"][1:].mean()) if len(o["time"]) > 1 else None)
    res[split] = r
    np.savez(os.path.join(a.out, f"{split}_estimates.npz"), index=idx, R=R_est, t=t_est, quat_allo=quat_allo,
             t_scaled=t_scaled)
    print(split, json.dumps(r), flush=True)
  json.dump(res, open(os.path.join(a.out, "esa.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
