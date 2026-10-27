"""LiePose head with an extra per-slice conditioning vector.

`cond_mode`:
  none     vanilla LiePose head (c ignored)
  pooled   c = foreground-pooled DINOv2 feature of the observed image (pose-independent control)
  compare  c = latent comparison between the world model's prediction at the noisy pose and the
           observed DINOv2 tokens (compare-in-the-loop)
"""

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from liepose.exp.model import FourierMlpBlock, MlpBlock, PosEmbed, _create_backbone, broadcast_batch, get_activ_fn


class CondHead(nn.Module):
  dim: int
  n_layers: int = 1
  block_type: nn.Module = MlpBlock
  activ: Callable = jax.nn.leaky_relu
  use_cond: bool = True
  # named + zero-initialised conditioning layers: the head starts as an exact vanilla LiePose head, so
  # it can be initialised from a trained LiePose checkpoint (layer names then match the upstream Head)
  named_cond: bool = False

  @nn.compact
  def __call__(self, x0, rt, t, c):
    activ = self.activ
    x0 = broadcast_batch(x0, rt.shape[0])
    t = PosEmbed(256, 256, True)(t)
    if self.use_cond:
      # the conditioning joins the time embedding, which modulates every block
      if self.named_cond:
        h = jax.nn.silu(nn.Dense(256, name="cond_in")(c))
        t = t + nn.Dense(256, name="cond_out", kernel_init=nn.initializers.zeros)(h)
      else:
        t = t + nn.Dense(256)(jax.nn.silu(nn.Dense(256)(c)))
    x = PosEmbed(256, 256, False)(rt)
    for _ in range(self.n_layers):
      x = self.block_type(256, activ)(self.block_type(256, activ)(x, x0), t)
    return nn.Dense(self.dim)(activ(x))


class CondModel(nn.Module):
  backbone: nn.Module
  head: nn.Module

  def __call__(self, img, rt, t, c):
    return self.head(self.backbone(img), rt, t, c)


def create_cond_model_fn(out_dim=6, in_dim=6, cond_dim=449, image_shape=(1, 224, 224, 3), resnet_depth=34,
                         mlp_layers=1, fourier_block=True, activ_fn="leaky_relu", use_cond=True, named_cond=False):
  def create(seed):
    backbone, backbone_params = _create_backbone(seed, dim=512, feat_dim=128, image_shape=list(image_shape),
                                                 resnet_depth=resnet_depth)
    feat = backbone.apply(backbone_params, jnp.ones(image_shape))
    head = CondHead(dim=out_dim, n_layers=mlp_layers, block_type=FourierMlpBlock if fourier_block else MlpBlock,
                    activ=get_activ_fn(activ_fn), use_cond=use_cond, named_cond=named_cond)
    head_params = head.init(seed, jnp.ones(feat.shape), jnp.ones([1, in_dim]), jnp.ones([1, 1]), jnp.ones([1, cond_dim]))
    params = FrozenDict({
      "params": {"backbone": backbone_params["params"], "head": head_params["params"]},
      "batch_stats": {"backbone": backbone_params["batch_stats"]},
    })
    return CondModel(backbone, head), params

  return create
