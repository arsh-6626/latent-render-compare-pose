from typing import Any, Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jaxlie import SE3, SO3

from liepose.utils import ops


class CEMConfig(NamedTuple):
  horizon: int = 8
  n_samples: int = 64
  n_elites: int = 8
  iterations: int = 3
  action_dim: int = 6
  action_min: float = -1.0
  action_max: float = 1.0
  init_std: float = 1.0
  min_std: float = 0.01
  momentum: float = 0.1
  w_latent: float = 1.0
  w_pose: float = 2.0
  w_action: float = 0.05


class MPCCEMPlanner:
  """JAX-Optimized Cross-Entropy Method (CEM) Planner for 3D Environments.

  Uses the trained D-JEPA predictor to roll out future 3D spatial representations
  and LiePose diffusion head to evaluate geometric trajectory costs.
  """

  def __init__(
      self,
      model_apply_fn: Callable,
      params: Dict[str, Any],
      config: CEMConfig = CEMConfig(),
  ):
    self.model_apply_fn = model_apply_fn
    self.params = params
    self.config = config

  def _rollout_single_trajectory(
      self,
      s_init: jnp.ndarray,
      action_seq: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Rolls out a single action sequence starting from initial state s_init (L, D) using jax.lax.scan.

    Args:
      s_init: (L, D) single initial JEPA context feature.
      action_seq: (H, action_dim) candidate action sequence.

    Returns:
      latents: (H, L, D) predicted future latent states.
      fused_feats: (H, D) pooled fused features for pose head.
    """
    s_curr = jnp.expand_dims(s_init, axis=0)  # (1, L, D)

    def _step_fn(carry_state, action_t):
      s_c = carry_state
      a_t = jnp.expand_dims(action_t, axis=0)  # (1, action_dim)

      # Predict next state via JEPA predictor
      s_next = self.model_apply_fn(
          self.params,
          s_c,
          a_t,
          method=lambda module, ctx, act: module.backbone.world_model.predict(
              ctx, act, deterministic=True
          ),
      )

      # Fused feature representation for pose head evaluation
      h_ctx = self.model_apply_fn(
          self.params,
          jnp.mean(s_c, axis=1),
          method=lambda module, x: module.backbone.ctx_dense(x),
      )
      h_pred = self.model_apply_fn(
          self.params,
          jnp.mean(s_next, axis=1),
          method=lambda module, x: module.backbone.pred_dense(x),
      )
      h_fused = self.model_apply_fn(
          self.params,
          jax.nn.leaky_relu(h_ctx + h_pred),
          method=lambda module, x: module.backbone.fusion_dense(x),
      )

      return s_next, (s_next[0], h_fused[0])

    _, (latents, fused_feats) = jax.lax.scan(_step_fn, s_curr, action_seq)
    return latents, fused_feats

  def _evaluate_trajectory_cost(
      self,
      latents: jnp.ndarray,
      fused_feats: jnp.ndarray,
      action_seq: jnp.ndarray,
      s_target: jnp.ndarray,
      target_pose_repr: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes trajectory cost given rollout latents and action sequence."""
    cfg = self.config
    H = action_seq.shape[0]

    # 1. Latent feature target distance cost
    s_target_tiled = jnp.tile(jnp.expand_dims(s_target, axis=0), (H, 1, 1))
    cost_latent = jnp.mean((latents - s_target_tiled) ** 2)

    # 2. Pose head score error cost
    dummy_t = jnp.zeros((H, 1), dtype=jnp.float32)
    dummy_rt = jnp.tile(jnp.expand_dims(target_pose_repr, axis=0), (H, 1))

    mu_pred = self.model_apply_fn(
        self.params,
        fused_feats,
        dummy_rt,
        dummy_t,
        method=lambda module, x, rt, t: module.head(x, rt, t),
    )
    cost_pose = jnp.mean(mu_pred**2)

    # 3. Action regularizer
    cost_action = jnp.mean(action_seq**2)

    return (
        cfg.w_latent * cost_latent
        + cfg.w_pose * cost_pose
        + cfg.w_action * cost_action
    )

  def plan(
      self,
      seed: jax.random.PRNGKey,
      img_obs: jnp.ndarray,
      img_target: jnp.ndarray,
      target_pose_repr: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Executes MPC CEM optimization for 3D environment action planning."""
    cfg = self.config

    # Encode context observation and target observation into JEPA latents
    s_0 = self.model_apply_fn(
        self.params,
        img_obs,
        method=lambda module, x: module.backbone.world_model.encode(
            x, deterministic=True
        ),
    )[0]  # (L, D)

    s_target = self.model_apply_fn(
        self.params,
        img_target,
        method=lambda module, x: module.backbone.world_model.encode(
            x, deterministic=True
        ),
    )[0]  # (L, D)

    # Gaussian action distribution parameters
    mean = jnp.zeros((cfg.horizon, cfg.action_dim), dtype=jnp.float32)
    std = jnp.ones((cfg.horizon, cfg.action_dim), dtype=jnp.float32) * cfg.init_std

    # Vectorized rollout over batch of candidate action sequences
    vmap_rollout = jax.vmap(
        lambda a_seq: self._rollout_single_trajectory(s_0, a_seq)
    )
    vmap_cost = jax.vmap(
        lambda lats, fused, a_seq: self._evaluate_trajectory_cost(
            lats, fused, a_seq, s_target, target_pose_repr
        )
    )

    costs_history = []
    key = seed

    for it in range(cfg.iterations):
      key, subkey = jax.random.split(key)

      # Sample candidate action sequences
      noise = jax.random.normal(
          subkey, shape=(cfg.n_samples, cfg.horizon, cfg.action_dim)
      )
      actions = mean + std * noise
      actions = jnp.clip(actions, cfg.action_min, cfg.action_max)

      # Parallel rollout & cost evaluation
      latents, fused_feats = vmap_rollout(actions)
      costs = vmap_cost(latents, fused_feats, actions)

      min_cost = jnp.min(costs)
      costs_history.append(min_cost)

      # Select top N_elites
      elite_indices = jnp.argsort(costs)[: cfg.n_elites]
      elite_actions = actions[elite_indices]

      # Refit Gaussian parameters with momentum
      new_mean = jnp.mean(elite_actions, axis=0)
      new_std = jnp.std(elite_actions, axis=0)
      new_std = jnp.maximum(new_std, cfg.min_std)

      mean = cfg.momentum * mean + (1.0 - cfg.momentum) * new_mean
      std = cfg.momentum * std + (1.0 - cfg.momentum) * new_std

    best_actions = mean
    best_latents, _ = self._rollout_single_trajectory(s_0, best_actions)
    costs_history = jnp.stack(costs_history)

    return best_actions, best_latents, costs_history
