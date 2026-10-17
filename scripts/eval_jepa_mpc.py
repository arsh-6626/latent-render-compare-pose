#!/usr/bin/env python3
"""Standalone Evaluation Script for D-JEPA + LiePose Joint Model & 3D Environment MPC CEM Planning."""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from liepose.data.jepa_env_dataset import JEPAEnvDataset
from liepose.exp.jepa_liepose.model import create_jepa_model_fn
from liepose.metrics import so3 as so3_metrics
from liepose.planning.mpc_cem import CEMConfig, MPCCEMPlanner


def parse_args():
  parser = argparse.ArgumentParser(
      description="Evaluate JEPA-conditioned LiePose & 3D MPC CEM Planning"
  )
  parser.add_argument("--img_res", type=int, default=224, help="Image resolution")
  parser.add_argument("--action_dim", type=int, default=6, help="Action dimension")
  parser.add_argument("--embed_dim", type=int, default=512, help="JEPA embedding dimension")
  parser.add_argument("--repr_type", type=str, default="tan", help="Pose representation type (tan / rotmat)")
  parser.add_argument("--cem_horizon", type=int, default=8, help="CEM planning horizon")
  parser.add_argument("--cem_samples", type=int, default=64, help="CEM candidate samples")
  parser.add_argument("--cem_elites", type=int, default=8, help="CEM elite samples")
  parser.add_argument("--cem_iterations", type=int, default=3, help="CEM optimization iterations")
  parser.add_argument("--eval_trials", type=int, default=3, help="Number of evaluation trials")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  return parser.parse_args()


def main():
  args = parse_args()
  print("=" * 70)
  print(" JEPA-Conditioned LiePose 3D Environment & MPC CEM Evaluator")
  print("=" * 70)
  repr_size = so3_metrics.get_repr_size(args.repr_type)
  print(f"Configuration: Image Res: {args.img_res} | Action Dim: {args.action_dim} | Embed Dim: {args.embed_dim} | Pose Repr: {args.repr_type} ({repr_size}D)")
  print(f"MPC CEM Specs: Horizon: {args.cem_horizon} | Samples: {args.cem_samples} | Elites: {args.cem_elites} | Iters: {args.cem_iterations}\n")

  # 1. Initialize Joint Model & Parameters
  model_fn = create_jepa_model_fn(
      out_dim=repr_size,
      in_dim=repr_size,
      action_dim=args.action_dim,
      image_shape=(1, args.img_res, args.img_res, 3),
      embed_dim=args.embed_dim,
  )
  seed = jax.random.PRNGKey(args.seed)
  model, params = model_fn(seed)
  print("✓ Initialized JAX D-JEPA World Model & LiePose Diffusion Head successfully.")

  # 2. Instantiate MPC CEM Planner
  cem_config = CEMConfig(
      horizon=args.cem_horizon,
      n_samples=args.cem_samples,
      n_elites=args.cem_elites,
      iterations=args.cem_iterations,
      action_dim=args.action_dim,
  )
  planner = MPCCEMPlanner(
      model_apply_fn=model.apply,
      params=params,
      config=cem_config,
  )
  print("✓ Initialized JAX-vectorized MPCCEMPlanner module.")

  # 3. Create Benchmark Dataset
  dataset = JEPAEnvDataset(
      batch_size=1,
      img_res=args.img_res,
      action_dim=args.action_dim,
      seed=args.seed + 100,
  )
  data_iter = iter(dataset)

  # 4. Run Planning Evaluation Trials
  print("\n--- Running 3D Environment MPC CEM Action Planning Trials ---")
  costs = []
  times = []

  eval_seed = jax.random.PRNGKey(args.seed + 1000)

  for trial in range(args.eval_trials):
    batch = next(data_iter)
    img_obs = jnp.array(batch["image"])
    img_target = jnp.array(batch["target_image"])
    
    # Convert ground-truth rotation quaternion to target representation (e.g. tangent space)
    rot_so3 = so3_metrics.as_lie(jnp.array(batch["rot_gt"][0]))
    target_pose_repr = so3_metrics.as_repr(rot_so3, args.repr_type)

    eval_seed, subkey = jax.random.split(eval_seed)
    start_t = time.time()
    best_actions, best_latents, cost_hist = planner.plan(
        subkey, img_obs, img_target, target_pose_repr
    )
    elapsed = time.time() - start_t
    final_cost = float(np.array(cost_hist[-1]))

    costs.append(final_cost)
    times.append(elapsed)

    print(f" Trial [{trial+1}/{args.eval_trials}]: Time={elapsed:.3f}s | Trajectory Cost={final_cost:.6f} | Final Actions Shape={best_actions.shape}")

  mean_cost = np.mean(costs)
  mean_time = np.mean(times)
  print("\n" + "=" * 70)
  print(" Evaluation Results Summary:")
  print(f" Mean MPC Trajectory Cost : {mean_cost:.6f}")
  print(f" Mean Planning Time       : {mean_time:.3f} s / trajectory")
  print("=" * 70)


if __name__ == "__main__":
  main()
