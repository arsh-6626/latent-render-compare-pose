"""T-LESS test of a trained LiePose checkpoint with LiePose's own test loop and BOP evaluator.

Testbed.run_test (uniform SE(3) start, 100-step DDIM, beta_skip, 1 sample per instance, final rt) over
tless-test_primesense (the images of test_targets_bop19.json; crops from the GT visible box, image x GT visible mask,
as LiePose's TestDataset), then data/bop/evaluator.Evaluator: BOP CSV + bop_toolkit AR (VSD, MSSD, MSPD) on the
bop19 targets. Only the VSD renderer backend differs from upstream (BOP_RENDERER=vispy; bop_renderer not installed).

  DATADIR=... BOP_RENDERER=vispy PYTHONPATH=src python scripts/tless/test_liepose.py \
      --exp $LOGS/experiments/tless-se3-hpc --step 400000 --out $LOGS/tless_test/liepose_se3_400k
"""

import argparse
import json
import os

from omegaconf import OmegaConf

from liepose.exp.cmp.bop_run import TestbedSE3
from liepose.utils.exploration import Options

# LiePose's test sampling options (liepose.exp.bop.run.build_explorations)
SAMPLE = dict(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=100, n_slices=1)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--exp", required=True)
  p.add_argument("--step", type=int, required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--no_ema", action="store_true", help="sample with the live weights instead of the EMA weights")
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  cfg = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(a.exp, "configs.yaml")), resolve=True))
  cfg.restore_path, cfg.restore_step, cfg.skip_train = None, a.step, True
  cfg.full_eval = True
  if a.no_ema:
    cfg.ema_enabled = False  # the EMA weights of our HPC checkpoint score far worse on the training loss
  tb = TestbedSE3(cfg, a.exp, setup=True)
  assert int(tb.state.step) == a.step, (int(tb.state.step), a.step)
  print(f"restored step {int(tb.state.step)}", flush=True)
  tb.run_test(Options(SAMPLE), a.out)
  tb.run_eval(Options(SAMPLE), a.out)
  json.dump({"exp": a.exp, "step": a.step, "sample": SAMPLE, "renderer": os.environ.get("BOP_RENDERER", "cpp"), "ema": not a.no_ema},
            open(os.path.join(a.out, "run.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
