import os
from typing import List

from liepose.exp.jepa_liepose.testbed import TestbedJEPA
from liepose.utils import ntfy_log
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Explorations, Options


def build_explorations() -> Explorations:
  explorations = Explorations()

  # Model & Architecture configs
  explorations.image_res = [224]
  explorations.repr_type = ["tan"]
  explorations.resnet_depth = [34]
  explorations.mlp_layers = [1]
  explorations.fourier_block = [True]
  explorations.activ_fn = ["leaky_relu"]
  
  # JEPA Specific Parameters
  explorations.action_dim = [6]
  explorations.embed_dim = [512]
  explorations.encoder_depth = [6]
  explorations.predictor_depth = [4]
  explorations.w_jepa = [1.0]
  explorations.w_pose = [1.0]
  explorations.tau_target = [0.99]

  # Training Hyperparameters
  explorations.learn_noise = [True]
  explorations.loss_name = ["chordal"]
  explorations.train_steps = [1000]  # Full training steps configurable
  explorations.batch_size = [8]
  explorations.n_slices = [256]
  explorations.ckpt_steps = [500]

  # Learning rate schedule
  explorations.init_lr = [1e-4]
  explorations.lr_decay_start = [0.5]
  explorations.lr_decay_steps = [0.5]
  explorations.lr_decay_rate = [0.1]
  explorations.end_lr = [1e-5]

  # EMA
  explorations.ema_enabled = [True]
  explorations.ema_steps = [5]
  explorations.ema_tau = [0.999]

  # Noise schedule
  explorations.noise_type = ["power"]
  explorations.noise_start = [1e-8]
  explorations.noise_end = [1.0]
  explorations.power = [3.0]
  explorations.timesteps = [100]

  # Execution flags
  explorations.restore_step = [None]
  explorations.restore_path = [None]
  explorations.skip_train = [False]
  explorations.skip_test = [False]
  explorations.seed = [42]

  # Sampling options
  sample_options = Explorations()
  explorations.sample_options = sample_options

  sample_options.use_ddpm = [False]
  sample_options.eta = [1.0]
  sample_options.beta_skip = [True]
  sample_options.steps = [100]
  sample_options.batch_size = [16]
  sample_options.n_slices = [1]

  return explorations


def run_experiment(a: Options, exp_path: str):
  sample_options_exps: Explorations = a.pop("sample_options")

  exp = TestbedJEPA(a, exp_path, setup=True)

  if not a.skip_train:
    save_config(a, os.path.join(exp_path, "configs.yaml"))
    exp.run_train_single()
    print(f"Joint Training Complete: {exp_path}")
  else:
    print("[Skip training]")

  if not a.skip_test:
    # Run 3D Environment Action Planning Evaluation via MPC CEM
    exp.run_mpc_cem_eval(n_eval_trials=5)
  else:
    print("[Skip testing]")


def main():
  a = get_args(exp_name="jepa-liepose-joint", desc="JEPA-Conditioned LiePose Experiment")
  root_path = a.root_path

  explorations = load_config_from_args(a, build_explorations())
  explorations_list: List[Options] = explorations.expand()

  for exploration in explorations_list:
    exp_name = exploration.to_string(includes=["timesteps", "repr_type", "seed"])
    exp_path = os.path.join(root_path, exp_name)
    run_experiment(exploration, exp_path)


if __name__ == "__main__":
  main()
