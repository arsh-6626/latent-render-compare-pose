import math
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

# ------------------------------------------------------------------------------
# Core Flax Neural Building Blocks for JEPA / D-JEPA
# ------------------------------------------------------------------------------

class PatchEmbed(nn.Module):
  """2D Image to Patch Embedding."""
  img_size: int = 224
  patch_size: int = 16
  in_chans: int = 3
  embed_dim: int = 768

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    # x shape: (B, H, W, C)
    B, H, W, C = x.shape
    num_patches = (H // self.patch_size) * (W // self.patch_size)
    
    # Convolutional patch projection
    x = nn.Conv(
        features=self.embed_dim,
        kernel_size=(self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding="VALID",
        name="proj",
    )(x)
    # Reshape from (B, H', W', embed_dim) to (B, L, embed_dim)
    x = x.reshape((B, -1, self.embed_dim))
    return x


class Attention(nn.Module):
  """Multi-head Self Attention Module in Flax."""
  dim: int
  num_heads: int = 8
  qkv_bias: bool = True
  attn_drop: float = 0.0
  proj_drop: float = 0.0

  @nn.compact
  def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    B, N, C = x.shape
    head_dim = self.dim // self.num_heads
    scale = head_dim**-0.5

    qkv = nn.Dense(features=self.dim * 3, use_bias=self.qkv_bias, name="qkv")(x)
    qkv = qkv.reshape((B, N, 3, self.num_heads, head_dim))
    qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * scale
    attn = jax.nn.softmax(attn, axis=-1)

    if not deterministic and self.attn_drop > 0.0:
      attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)

    x = jnp.transpose(attn @ v, (0, 2, 1, 3)).reshape((B, N, C))
    x = nn.Dense(features=self.dim, name="proj")(x)

    if not deterministic and self.proj_drop > 0.0:
      x = nn.Dropout(rate=self.proj_drop)(x, deterministic=deterministic)
    return x


class Mlp(nn.Module):
  """MLP Block for Transformer."""
  in_features: int
  hidden_features: Optional[int] = None
  out_features: Optional[int] = None
  act_layer: Callable = jax.nn.gelu
  drop: float = 0.0

  @nn.compact
  def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    h_dim = self.hidden_features or self.in_features * 4
    out_dim = self.out_features or self.in_features

    x = nn.Dense(features=h_dim, name="fc1")(x)
    x = self.act_layer(x)
    if not deterministic and self.drop > 0.0:
      x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
    x = nn.Dense(features=out_dim, name="fc2")(x)
    if not deterministic and self.drop > 0.0:
      x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
    return x


class TransformerBlock(nn.Module):
  """Transformer Encoder / Predictor Block."""
  dim: int
  num_heads: int
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  drop: float = 0.0
  attn_drop: float = 0.0

  @nn.compact
  def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    norm1 = nn.LayerNorm(name="norm1")(x)
    attn_out = Attention(
        dim=self.dim,
        num_heads=self.num_heads,
        qkv_bias=self.qkv_bias,
        attn_drop=self.attn_drop,
        proj_drop=self.drop,
        name="attn",
    )(norm1, deterministic=deterministic)
    x = x + attn_out

    norm2 = nn.LayerNorm(name="norm2")(x)
    mlp_hidden_dim = int(self.dim * self.mlp_ratio)
    mlp_out = Mlp(
        in_features=self.dim,
        hidden_features=mlp_hidden_dim,
        out_features=self.dim,
        drop=self.drop,
        name="mlp",
    )(norm2, deterministic=deterministic)
    x = x + mlp_out
    return x


# ------------------------------------------------------------------------------
# JEPA Encoder & Predictor Submodules
# ------------------------------------------------------------------------------

class JEPAEncoder(nn.Module):
  """JEPA Vision Transformer Context & Target Encoder in JAX."""
  img_size: int = 224
  patch_size: int = 16
  in_chans: int = 3
  embed_dim: int = 512
  depth: int = 6
  num_heads: int = 8
  mlp_ratio: float = 4.0
  buffer_size: int = 16

  @nn.compact
  def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    # x shape: (B, H, W, C)
    B = x.shape[0]
    
    # Patch embedding
    patches = PatchEmbed(
        img_size=self.img_size,
        patch_size=self.patch_size,
        in_chans=self.in_chans,
        embed_dim=self.embed_dim,
        name="patch_embed",
    )(x)
    num_patches = patches.shape[1]

    # Learned positional embeddings
    pos_embed = self.param(
        "pos_embed",
        nn.initializers.normal(stddev=0.02),
        (1, num_patches, self.embed_dim),
    )
    patches = patches + pos_embed

    # Learned buffer / memory tokens (as in D-JEPA)
    buffer = self.param(
        "buffer",
        nn.initializers.normal(stddev=0.02),
        (1, self.buffer_size, self.embed_dim),
    )
    buffer_tiled = jnp.tile(buffer, (B, 1, 1))

    # Prepend buffer tokens to sequence
    x_tokens = jnp.concatenate([buffer_tiled, patches], axis=1)

    # Process through Transformer blocks
    for i in range(self.depth):
      x_tokens = TransformerBlock(
          dim=self.embed_dim,
          num_heads=self.num_heads,
          mlp_ratio=self.mlp_ratio,
          name=f"block_{i}",
      )(x_tokens, deterministic=deterministic)

    x_tokens = nn.LayerNorm(name="norm")(x_tokens)
    return x_tokens


class JEPAPredictor(nn.Module):
  """JEPA Action-Conditioned Spatial & 3D Feature Predictor in JAX."""
  embed_dim: int = 512
  predictor_embed_dim: int = 512
  action_dim: int = 6
  depth: int = 4
  num_heads: int = 8
  mlp_ratio: float = 4.0

  @nn.compact
  def __call__(
      self,
      context_feats: jnp.ndarray,
      action: jnp.ndarray,
      deterministic: bool = True,
  ) -> jnp.ndarray:
    """Predicts future state representation given context features and action.

    Args:
      context_feats: (B, L, embed_dim) from JEPAEncoder.
      action: (B, action_dim) 3D environment action vector (e.g. SE(3) control).
      deterministic: boolean flag for dropout.

    Returns:
      predicted_feats: (B, L, predictor_embed_dim)
    """
    B, L, C = context_feats.shape

    # Project context features to predictor embedding dimension if needed
    x = nn.Dense(features=self.predictor_embed_dim, name="proj_in")(context_feats)

    # Embed action vector and broadcast across spatial tokens
    action_emb = nn.Dense(features=self.predictor_embed_dim, name="action_proj")(
        jax.nn.silu(nn.Dense(features=self.predictor_embed_dim)(action))
    )
    action_emb = jnp.expand_dims(action_emb, axis=1)  # (B, 1, predictor_embed_dim)

    # Add action conditioning to tokens
    x = x + action_emb

    # Predictor Transformer blocks
    for i in range(self.depth):
      x = TransformerBlock(
          dim=self.predictor_embed_dim,
          num_heads=self.num_heads,
          mlp_ratio=self.mlp_ratio,
          name=f"pred_block_{i}",
      )(x, deterministic=deterministic)

    x = nn.LayerNorm(name="pred_norm")(x)
    return x


# ------------------------------------------------------------------------------
# Full JAX D-JEPA World Model Engine
# ------------------------------------------------------------------------------

class DJEPAWorldModel(nn.Module):
  """Complete JAX D-JEPA Joint Embedding Predictive Architecture.

  Combines context encoder, action-conditioned predictor, and latent flow matching
  denoising head to provide 3D geometry-aware spatial predictive features.
  """
  img_size: int = 224
  patch_size: int = 16
  in_chans: int = 3
  encoder_embed_dim: int = 512
  predictor_embed_dim: int = 512
  action_dim: int = 6
  encoder_depth: int = 6
  predictor_depth: int = 4
  num_heads: int = 8
  buffer_size: int = 16

  def setup(self):
    self.encoder = JEPAEncoder(
        img_size=self.img_size,
        patch_size=self.patch_size,
        in_chans=self.in_chans,
        embed_dim=self.encoder_embed_dim,
        depth=self.encoder_depth,
        num_heads=self.num_heads,
        buffer_size=self.buffer_size,
        name="encoder",
    )
    self.predictor = JEPAPredictor(
        embed_dim=self.encoder_embed_dim,
        predictor_embed_dim=self.predictor_embed_dim,
        action_dim=self.action_dim,
        depth=self.predictor_depth,
        num_heads=self.num_heads,
        name="predictor",
    )
    # D-JEPA Flow-matching / Denoising predictor for latent tokens
    self.flow_head = nn.Dense(
        features=self.encoder_embed_dim, name="flow_head"
    )

  def encode(self, img: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    """Encodes observation image into latent state representations."""
    return self.encoder(img, deterministic=deterministic)

  def predict(
      self,
      context_feats: jnp.ndarray,
      action: jnp.ndarray,
      deterministic: bool = True,
  ) -> jnp.ndarray:
    """Predicts target latent state given context features and action."""
    return self.predictor(context_feats, action, deterministic=deterministic)

  def __call__(
      self,
      img_t: jnp.ndarray,
      action: jnp.ndarray,
      deterministic: bool = True,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass computing context features and predicted future features."""
    s_t = self.encode(img_t, deterministic=deterministic)
    s_next_pred = self.predict(s_t, action, deterministic=deterministic)
    return s_t, s_next_pred


# ------------------------------------------------------------------------------
# Loss Functions for JEPA & D-JEPA
# ------------------------------------------------------------------------------

def jepa_smooth_l1_loss(
    pred_latents: jnp.ndarray, target_latents: jnp.ndarray, beta: float = 1.0
) -> jnp.ndarray:
  """Smooth L1 Loss between predicted latents and EMA target latents."""
  diff = pred_latents - target_latents
  abs_diff = jnp.abs(diff)
  smooth_l1 = jnp.where(
      abs_diff < beta, 0.5 * (abs_diff**2) / beta, abs_diff - 0.5 * beta
  )
  return jnp.mean(smooth_l1)


def djepa_flow_matching_loss(
    pred_flow: jnp.ndarray, target_latents: jnp.ndarray, t: jnp.ndarray
) -> jnp.ndarray:
  """Flow Matching loss for D-JEPA latent generation."""
  # t in [0, 1]
  # target velocity v = target - noise
  target_velocity = target_latents
  loss = jnp.mean((pred_flow - target_velocity) ** 2)
  return loss
