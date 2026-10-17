from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.core.frozen_dict import unfreeze

from liepose.exp.model import FourierMlpBlock, Head, MlpBlock, PosEmbed, broadcast_batch, get_activ_fn
from liepose.models.djepa_jax import DJEPAWorldModel, JEPAEncoder, JEPAPredictor


class JEPAConditionedBackbone(nn.Module):
  """Backbone that extracts JEPA context and predictor features to condition pose diffusion.

  Integrates the D-JEPA encoder and predictor features into a unified 3D-aware latent representation.
  """
  dim: int = 512
  feat_dim: int = 128
  img_size: int = 224
  patch_size: int = 16
  action_dim: int = 6
  encoder_depth: int = 6
  predictor_depth: int = 4
  num_heads: int = 8
  buffer_size: int = 16
  activ: Callable = jax.nn.leaky_relu

  def setup(self):
    self.world_model = DJEPAWorldModel(
        img_size=self.img_size,
        patch_size=self.patch_size,
        in_chans=3,
        encoder_embed_dim=self.dim,
        predictor_embed_dim=self.dim,
        action_dim=self.action_dim,
        encoder_depth=self.encoder_depth,
        predictor_depth=self.predictor_depth,
        num_heads=self.num_heads,
        buffer_size=self.buffer_size,
        name="world_model",
    )
    self.ctx_dense = nn.Dense(self.dim, name="ctx_dense")
    self.pred_dense = nn.Dense(self.dim, name="pred_dense")
    self.fusion_dense = nn.Dense(self.dim, name="fusion_dense")

  def __call__(
      self, img: jnp.ndarray, action: jnp.ndarray, deterministic: bool = True
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes fused JEPA features and individual token representations.

    Args:
      img: (B, H, W, C) observation image.
      action: (B, action_dim) action vector (e.g. ego-motion / SE(3) shift).
      deterministic: boolean flag for dropout.

    Returns:
      h_fused: (B, dim) fused 3D geometry-aware representation for pose head.
      s_t: (B, L, dim) context tokens from JEPA encoder.
      s_pred: (B, L, dim) predicted future tokens from JEPA predictor.
    """
    activ = self.activ
    # 1. JEPA Encoder context representation
    s_t = self.world_model.encode(img, deterministic=deterministic)  # (B, L, dim)
    
    # 2. JEPA Predictor future representation
    s_pred = self.world_model.predict(s_t, action, deterministic=deterministic)  # (B, L, dim)

    # 3. Spatial pooling over token sequence (excluding buffer tokens if desired, or mean over all)
    s_t_pooled = jnp.mean(s_t, axis=1)  # (B, dim)
    s_pred_pooled = jnp.mean(s_pred, axis=1)  # (B, dim)

    # 4. Feature projection & non-linear fusion
    h_ctx = self.ctx_dense(s_t_pooled)
    h_pred = self.pred_dense(s_pred_pooled)
    
    h_fused = self.fusion_dense(activ(h_ctx + h_pred))
    return h_fused, s_t, s_pred


class JEPAConditionedLiePoseModel(nn.Module):
  """Joint LiePose Diffusion and JEPA World Model Architecture in JAX."""
  backbone: JEPAConditionedBackbone
  head: Head

  def __call__(
      self,
      img: jnp.ndarray,
      action: jnp.ndarray,
      rt: jnp.ndarray,
      t: jnp.ndarray,
      deterministic: bool = True,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward pass computing pose noise prediction mu and JEPA latents.

    Args:
      img: (B, H, W, 3) observation image.
      action: (B, action_dim) 3D environment action vector.
      rt: (BS, in_dim) noisy pose Lie group tangent vector / representation.
      t: (BS, 1) diffusion timesteps.
      deterministic: boolean flag.

    Returns:
      mu: (BS, out_dim) predicted pose tangent vector / noise.
      s_t: (B, L, dim) JEPA context features.
      s_pred: (B, L, dim) JEPA predicted future features.
    """
    h_fused, s_t, s_pred = self.backbone(img, action, deterministic=deterministic)
    mu = self.head(h_fused, rt, t)
    return mu, s_t, s_pred


def create_jepa_model_fn(
    out_dim: int = 6,
    in_dim: int = 6,
    action_dim: int = 6,
    image_shape: Sequence[int] = (1, 224, 224, 3),
    mlp_layers: int = 1,
    fourier_block: bool = True,
    activ_fn: str = "leaky_relu",
    encoder_depth: int = 6,
    predictor_depth: int = 4,
    embed_dim: int = 512,
):
  """Factory function to build the JEPA-conditioned LiePose model and initialize parameters."""
  def create_model_and_params(seed: jax.random.PRNGKey):
    backbone = JEPAConditionedBackbone(
        dim=embed_dim,
        feat_dim=128,
        img_size=image_shape[1],
        patch_size=16,
        action_dim=action_dim,
        encoder_depth=encoder_depth,
        predictor_depth=predictor_depth,
        num_heads=8,
        buffer_size=16,
        activ=get_activ_fn(activ_fn),
    )

    block_type = FourierMlpBlock if fourier_block else MlpBlock
    head = Head(
        dim=out_dim,
        n_layers=mlp_layers,
        block_type=block_type,
        activ=get_activ_fn(activ_fn),
    )

    model = JEPAConditionedLiePoseModel(backbone=backbone, head=head)

    # Sample dummy inputs for initialization
    dummy_img = jnp.ones(image_shape, dtype=jnp.float32)
    dummy_action = jnp.zeros((image_shape[0], action_dim), dtype=jnp.float32)
    dummy_rt = jnp.ones((image_shape[0], in_dim), dtype=jnp.float32)
    dummy_t = jnp.ones((image_shape[0], 1), dtype=jnp.float32)

    params = model.init(
        seed, dummy_img, dummy_action, dummy_rt, dummy_t, deterministic=True
    )
    return model, params

  return create_model_and_params
