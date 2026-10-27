"""Upstream LiePose T-LESS training, made resumable across 24 h cluster jobs.

Same config as liepose.exp.bop.run; training continues from the latest checkpoint up to
train_steps (Testbed.run_train_single). Testing/evaluation are skipped here (our own sampling
and metric scripts are run on the saved checkpoint).

  DATADIR=... CACHE_ROOT_PATH=... python -m liepose.exp.cmp.bop_run --root_path $LOGS/tless-se3 "lie_type=[se3]"
"""

import os
from typing import List

from liepose.exp.bop import run as bop_run
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Options


class ResumableMixin:
  """Kept for the classes that list it: Testbed.run_train_single itself now continues from the restored step
  up to train_steps, so shortening the loop here as well would stop resumed runs early."""


class TestbedSE3(ResumableMixin, bop_run.TestbedSE3):
  pass


def main():
  a = get_args(exp_name="bop-tless-score-flat", desc="LiePose T-LESS (resumable)")
  explorations = load_config_from_args(a, bop_run.build_explorations())
  explorations.skip_test = [True]
  explorations.skip_eval = [True]
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
