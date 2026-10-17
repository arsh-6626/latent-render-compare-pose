"""LiePose SE(3) on SPEED+ (train on synthetic, test on synthetic val / lightbox / sunlamp), resumable across
24 h cluster jobs. Same model and schedule as the T-LESS config (liepose.exp.bop.run); only the dataset differs.
Testing/evaluation are skipped here (our own sampling and metric scripts run on the saved checkpoint).

  DATADIR=... python -m liepose.exp.speedplus.run --root_path $LOGS/speedplus-se3 "lie_type=[se3]"
"""

import os

from liepose.data.speedplus.dataset import TestDataset, TrainDataset
from liepose.exp.bop import run as bop_run
from liepose.exp.cmp.bop_run import ResumableMixin
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Options

# standardisation of the scaled translation, from 3000 augmented training crops
TRAN_SCALE = [0.216053, 0.213263, 3.965901]
TRAN_OFFSET = [-0.02472, 0.002721, 5.666584]

AUGMENT = dict(random_color_prob=0.8, random_bg_prob=0.0, random_bbox_dzi=True, dzi_pad_scale=1.5,
               dzi_scale_ratio=0.25, dzi_shift_ratio=0.25, dzi_type="uniform")


class SpeedPlusMixin:
  def create_train_dataset(self):
    return TrainDataset(self.a.train_datasets, batch_size=self.a.batch_size, seed=self.a.seed,
                        num_workers=self.a.num_workers, input_res=self.a.image_res, tran_scale=self.a.tran_scale,
                        tran_offset=self.a.tran_offset, augment=AUGMENT)

  def create_test_dataset(self):
    return TestDataset(self.a.test_datasets, batch_size=self.opt.batch_size, input_res=self.a.image_res,
                       tran_scale=self.a.tran_scale, tran_offset=self.a.tran_offset, augment=AUGMENT)


class TestbedSE3(ResumableMixin, SpeedPlusMixin, bop_run.TestbedSE3):
  pass


def build_explorations():
  e = bop_run.build_explorations()
  e.train_datasets = ["train"]  # SPEED+ split names (liepose.data.speedplus.dataset.SPLITS)
  e.test_datasets = ["val"]
  e.tran_scale = [TRAN_SCALE]
  e.tran_offset = [TRAN_OFFSET]
  e.use_mask = [False]
  e.skip_test = [True]
  e.skip_eval = [True]
  return e


def main():
  a = get_args(exp_name="speedplus-se3", desc="LiePose SPEED+ (resumable)")
  explorations = load_config_from_args(a, build_explorations())
  for opt in explorations.expand():
    opt: Options
    opt.pop("sample_options", None)
    exp_path = os.path.join(a.root_path, opt.to_string(includes=["lie_type", "timesteps", "repr_type", "seed"]))
    assert opt.lie_type == "se3"
    tb = TestbedSE3(opt, exp_path, setup=True)
    save_config(opt, os.path.join(exp_path, "configs.yaml"))
    tb.run_train()


if __name__ == "__main__":
  main()
