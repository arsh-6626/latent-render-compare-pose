import os
import time
from typing import Any, Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
try:
  from flax.metrics import tensorboard
except Exception:
  class _DummySummaryWriter:
    def __init__(self, *args, **kwargs): pass
    def scalar(self, *args, **kwargs): pass
    def flush(self): pass
  class _DummyTensorboardModule:
    SummaryWriter = _DummySummaryWriter
  tensorboard = _DummyTensorboardModule()
from flax.training import checkpoints, train_state
from jaxlie import SO3
from tqdm import tqdm

from liepose.data.jepa_env_dataset import JEPAEnvDataset
from liepose.data.utils import set_seed
from liepose.dist import NormalSO3
from liepose.exp.jepa_liepose.model import create_jepa_model_fn
from liepose.exp.testbed import Testbed as BaseTestbed, restore_state, save_checkpoint
from liepose.metrics import so3 as so3_metrics
from liepose.models.djepa_jax import jepa_smooth_l1_loss
from liepose.planning.mpc_cem import CEMConfig, MPCCEMPlanner
from liepose.utils import ops
from liepose.utils.exploration import Options


class TrainStateJEPA(train_state.TrainState):
  batch_stats: Any
  params_ema: Any = None
  params_target_jepa: Any = None


class TestbedJEPA(BaseTestbed):
  Lie = SO3
  LieDist = NormalSO3
  lie_metrics = so3_metrics

  def create_model(self):
    lie_metrics = self.lie_metrics
    size = self.a.image_res
    repr_size = lie_metrics.get_repr_size(self.a.repr_type)
    action_dim = getattr(self.a, "action_dim", 6)
    embed_dim = getattr(self.a, "embed_dim", 512)

    model_fn = create_jepa_model_fn(
        out_dim=repr_size,
        in_dim=repr_size,
        action_dim=action_dim,
        image_shape=[1, size, size, 3],
        mlp_layers=self.a.mlp_layers,
        fourier_block=self.a.fourier_block,
        activ_fn=self.a.activ_fn,
        encoder_depth=getattr(self.a, "encoder_depth", 6),
        predictor_depth=getattr(self.a, "predictor_depth", 4),
        embed_dim=embed_dim,
    )
    model, params = model_fn(jax.random.PRNGKey(self.a.seed))
    return model, params

  def create_state(self, params, optim) -> TrainStateJEPA:
    assert self.model is not None
    # Initialize target EMA encoder parameters with context encoder weights
    target_params = params["params"]["backbone"]["world_model"]["encoder"]
    return TrainStateJEPA.create(
        apply_fn=self.model.apply,
        params=params["params"],
        tx=optim,
        batch_stats=params.get("batch_stats", {}),
        params_ema=params["params"],
        params_target_jepa=target_params,
    )

  def create_train_dataset(self):
    return JEPAEnvDataset(
        batch_size=self.a.batch_size,
        img_res=self.a.image_res,
        action_dim=getattr(self.a, "action_dim", 6),
        seed=self.a.seed,
    )

  def create_test_dataset(self):
    return JEPAEnvDataset(
        batch_size=getattr(self.opt, "batch_size", 16),
        img_res=self.a.image_res,
        action_dim=getattr(self.a, "action_dim", 6),
        seed=self.a.seed + 100,
    )

  def create_test_evaluator(self, dataset):
    # Evaluator stub for JEPA pose & planning metrics
    class EvaluatorStub:
      def __init__(self):
        self.results = []
      def process(self, batch, outputs):
        self.results.append(outputs)
      def summarize(self):
        print("\n--- Joint JEPA + LiePose Evaluation Summary ---")
        print(f"Evaluated {len(self.results)} batches successfully.")
    return EvaluatorStub()

  def create_get_batch_test_fn(self):
    sample_train_fn = self.create_sample_train_fn()

    def get_batch_test(seed, batch, n_slices=256):
      img = jnp.array(batch["image"], dtype=jnp.float32)
      action = jnp.array(batch["action"], dtype=jnp.float32)
      target_img = jnp.array(batch["target_image"], dtype=jnp.float32)
      rot = jnp.array(batch["rot_gt"], dtype=jnp.float32)

      batch_size = img.shape[0]
      seeds = jax.random.split(seed, batch_size)
      sampled = jax.vmap(sample_train_fn)(seeds, rot)

      return {
          "img": img,
          "action": action,
          "target_img": target_img,
          "rt": sampled["rt"],
          "t": sampled["t"],
          "zt": sampled["zt"],
          "r0": sampled["r0"],
      }

    return get_batch_test

  def create_sample_train_fn(self):
    Lie = self.Lie
    LieDist = self.LieDist
    lie_metrics = self.lie_metrics
    noise_schedule = self.noise_schedule

    def sample_train_fn(seed, rot):
      key1, key2 = jax.random.split(seed, 2)
      t = jax.random.randint(
          key2, shape=(), dtype=jnp.int32, minval=0, maxval=noise_schedule.timesteps
      )
      r0 = lie_metrics.as_lie(rot)
      zt = LieDist._sample_unit(key1, n=())
      rt = ops.add(r0, Lie.exp(noise_schedule.sqrt_alphas[t] * zt))
      rt = lie_metrics.as_repr(rt, self.a.repr_type)
      zt = lie_metrics.as_repr(zt, self.a.repr_type)
      r0 = lie_metrics.as_repr(r0, self.a.repr_type)
      return {"rt": rt, "t": t, "zt": zt, "r0": r0}

    return sample_train_fn

  def create_get_batch_train_fn(self):
    sample_train_fn = self.create_sample_train_fn()

    def get_batch_train(seed, batch, n_slices=256):
      img = jnp.array(batch["image"], dtype=jnp.float32)
      action = jnp.array(batch["action"], dtype=jnp.float32)
      target_img = jnp.array(batch["target_image"], dtype=jnp.float32)
      rot = jnp.array(batch["rot_gt"], dtype=jnp.float32)

      batch_size = img.shape[0]
      seeds = jax.random.split(seed, batch_size)

      # Sample diffusion noise and time for each batch item
      sampled = jax.vmap(sample_train_fn)(seeds, rot)
      
      return {
          "img": img,
          "action": action,
          "target_img": target_img,
          "rt": sampled["rt"],
          "t": sampled["t"],
          "zt": sampled["zt"],
          "r0": sampled["r0"],
      }

    return get_batch_train

  def create_update_fn(self) -> Callable:
    lie_metrics = self.lie_metrics
    learn_noise = self.a.learn_noise
    loss_name = self.a.loss_name
    w_jepa = getattr(self.a, "w_jepa", 1.0)
    w_pose = getattr(self.a, "w_pose", 1.0)
    tau_target = getattr(self.a, "tau_target", 0.99)

    def update_fn(state: TrainStateJEPA, batch: Dict[str, jnp.ndarray]):
      def loss_fn(params):
        t = batch["t"].reshape((-1, 1))
        
        # 1. Forward model (Pose Head + JEPA Context Encoder + JEPA Predictor)
        mu, s_t, s_pred = state.apply_fn(
            {"params": params},
            batch["img"],
            batch["action"],
            batch["rt"],
            t,
            deterministic=False,
        )

        # 2. Forward target EMA JEPA encoder for target latent generation
        # Compute target latents without gradient
        target_encoder_params = state.params_target_jepa
        target_params_dict = {
            "params": {
                "backbone": {
                    "world_model": {
                        "encoder": target_encoder_params
                    }
                }
            }
        }
        target_latents = state.apply_fn(
            target_params_dict,
            batch["target_img"],
            deterministic=True,
            method=lambda module, x, deterministic: module.backbone.world_model.encoder(
                x, deterministic=deterministic
            ),
        )
        target_latents = jax.lax.stop_gradient(target_latents)

        # 3. Pose diffusion loss
        ta = batch["zt"] if learn_noise else batch["r0"]
        pose_loss = jnp.mean(jax.vmap(lie_metrics.get_distance_fn(loss_name))(ta, mu))

        # 4. JEPA predictor loss
        jepa_loss = jepa_smooth_l1_loss(s_pred, target_latents)

        total_loss = w_pose * pose_loss + w_jepa * jepa_loss
        return total_loss, (pose_loss, jepa_loss)

      (total_loss, (pose_loss, jepa_loss)), grads = jax.value_and_grad(
          loss_fn, has_aux=True
      )(state.params)
      
      new_state = state.apply_gradients(grads=grads)

      # 5. Update JEPA Target Encoder EMA parameters
      new_encoder_params = new_state.params["backbone"]["world_model"]["encoder"]
      updated_target_jepa = jax.tree_util.tree_map(
          lambda p_target, p_enc: tau_target * p_target + (1.0 - tau_target) * p_enc,
          state.params_target_jepa,
          new_encoder_params,
      )
      new_state = new_state.replace(params_target_jepa=updated_target_jepa)

      return total_loss, pose_loss, jepa_loss, new_state

    return update_fn

  def run_train_single(self):
    self.assert_setup()
    print("\n--- Starting Joint D-JEPA World Model & LiePose Diffusion Training ---")

    set_seed(self.a.seed)
    rng_seq = hk.PRNGSequence(self.a.seed)
    train_dataset = self.create_train_dataset()
    train_iter = iter(train_dataset)

    summary_writer = tensorboard.SummaryWriter(self.exp_path)
    get_batch_train_fn = self.create_get_batch_train_fn()
    jit_update_fn = jax.jit(self.create_update_fn())

    ckpt_dir = os.path.join(self.exp_path, "ckpt")
    pbar = tqdm(range(self.a.train_steps), leave=True, desc="Train Joint Model")

    for step in pbar:
      batch = get_batch_train_fn(next(rng_seq), next(train_iter))
      total_loss, pose_loss, jepa_loss, self.state = jit_update_fn(self.state, batch)
      
      total_loss_val = np.array(total_loss).item()
      pose_loss_val = np.array(pose_loss).item()
      jepa_loss_val = np.array(jepa_loss).item()
      global_step = self.state.step

      if step % 5 == 0:
        pbar.set_postfix({
            "tot_loss": f"{total_loss_val:.4f}",
            "pose": f"{pose_loss_val:.4f}",
            "jepa": f"{jepa_loss_val:.4f}",
        })
        summary_writer.scalar("train/total_loss", total_loss_val, global_step)
        summary_writer.scalar("train/pose_loss", pose_loss_val, global_step)
        summary_writer.scalar("train/jepa_loss", jepa_loss_val, global_step)

      if (step + 1) % self.a.ckpt_steps == 0:
        save_checkpoint(ckpt_dir=ckpt_dir, target=self.state, step=global_step)

    save_checkpoint(ckpt_dir=ckpt_dir, target=self.state, step=global_step + 1)
    summary_writer.flush()
    return self

  def run_mpc_cem_eval(self, n_eval_trials: int = 5):
    """Evaluates 3D environment action planning performance using JAX MPC CEM planner."""
    print("\n--- Running 3D Environment Action Planning Evaluation (MPC CEM) ---")
    planner = MPCCEMPlanner(
        model_apply_fn=self.model.apply,
        params={"params": self.state.params},
        config=CEMConfig(horizon=8, n_samples=128, n_elites=16, iterations=5),
    )

    test_ds = self.create_test_dataset()
    test_iter = iter(test_ds)
    
    seed = jax.random.PRNGKey(self.a.seed + 999)
    costs_all = []

    for trial in range(n_eval_trials):
      batch = next(test_iter)
      img_obs = jnp.array(batch["image"][:1])
      img_target = jnp.array(batch["target_image"][:1])
      target_pose = jnp.array(batch["rot_gt"][0])  # dummy pose target

      seed, subkey = jax.random.split(seed)
      start_t = time.time()
      best_actions, best_latents, cost_history = planner.plan(
          subkey, img_obs, img_target, target_pose
      )
      planning_time = time.time() - start_t
      final_cost = np.array(cost_history[-1]).item()
      costs_all.append(final_cost)

      print(
          f"Trial {trial+1}/{n_eval_trials} | Planning Time: {planning_time:.3f}s | "
          f"Final Trajectory Cost: {final_cost:.6f}"
      )

    mean_cost = np.mean(costs_all)
    print(f"Mean MPC CEM Planning Cost across {n_eval_trials} trials: {mean_cost:.6f}\n")
    return mean_cost
