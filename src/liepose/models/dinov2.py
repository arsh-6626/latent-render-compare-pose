"""DINOv2 ViT-S/14 in flax, for training the backbone inside the JAX denoisers (the frozen token caches come from the
torch hub model; scripts/speedplus/convert_dinov2.py converts its weights).

Same computation as dinov2.models.vision_transformer.DinoVisionTransformer.forward_features at 224 x 224 (16 x 16
patches, no registers): patch conv, cls token, position embedding (the hub model's bicubic interpolation to 16 x 16 is
done once in torch by the converter and stored as the parameter), 12 pre-norm blocks with LayerScale, final LayerNorm
(eps 1e-6). Returns the normalised patch tokens (N, 256, 384).
"""

import flax.linen as nn
import jax.numpy as jnp

MEAN = jnp.array([0.485, 0.456, 0.406])
STD = jnp.array([0.229, 0.224, 0.225])


class Block(nn.Module):
  dim: int = 384
  heads: int = 6

  @nn.compact
  def __call__(self, x):
    N, L, D = x.shape
    h = nn.LayerNorm(epsilon=1e-6, name="norm1")(x)
    qkv = nn.Dense(3 * D, name="qkv")(h).reshape(N, L, 3, self.heads, D // self.heads)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    a = jnp.einsum("nqhd,nkhd->nhqk", q, k) * (D // self.heads) ** -0.5
    h = jnp.einsum("nhqk,nkhd->nqhd", nn.softmax(a, -1), v).reshape(N, L, D)
    x = x + self.param("ls1", nn.initializers.ones, (D,)) * nn.Dense(D, name="proj")(h)
    h = nn.LayerNorm(epsilon=1e-6, name="norm2")(x)
    h = nn.Dense(D, name="fc2")(nn.gelu(nn.Dense(4 * D, name="fc1")(h), approximate=False))
    return x + self.param("ls2", nn.initializers.ones, (D,)) * h


class DinoV2(nn.Module):
  dim: int = 384
  depth: int = 12
  heads: int = 6
  patch: int = 14
  grid: int = 16

  @nn.compact
  def __call__(self, image):
    """image (N, 224, 224, 3), 0-255 -> patch tokens (N, grid * grid, dim)"""
    x = (image / 255.0 - MEAN) / STD
    x = nn.Conv(self.dim, (self.patch, self.patch), strides=self.patch, padding="VALID", name="patch")(x)
    x = x.reshape(x.shape[0], -1, self.dim)
    cls = jnp.broadcast_to(self.param("cls", nn.initializers.zeros, (1, 1, self.dim)), (x.shape[0], 1, self.dim))
    x = jnp.concatenate([cls, x], 1) + self.param("pos", nn.initializers.zeros, (1, 1 + self.grid ** 2, self.dim))
    for i in range(self.depth):
      x = Block(self.dim, self.heads, name=f"block_{i}")(x)
    return nn.LayerNorm(epsilon=1e-6, name="norm")(x)[:, 1:]


def pool2(z, grid=16):
  """(N, grid^2, D) -> (N, (grid/2)^2, D): the caches' 2 x 2 average pooling (encode_crops.py)"""
  N, _, D = z.shape
  return z.reshape(N, grid // 2, 2, grid // 2, 2, D).mean((2, 4)).reshape(N, -1, D)
