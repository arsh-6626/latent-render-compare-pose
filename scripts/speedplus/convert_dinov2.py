"""Torch hub DINOv2 ViT-S/14 -> flax msgpack for liepose.models.dinov2.DinoV2 (torch env, e.g. `jepa`; run outside the
hub cache directory). The position embedding is interpolated to 16 x 16 here with the hub model's own code.
--check: writes reference fp32 patch tokens of a few crop images next to the output, for the JAX-side comparison.

  python scripts/speedplus/convert_dinov2.py --out $DATADIR/dinov2_vits14_224.msgpack --check $DATADIR/speedplus_crops/val/shard_0000.npz
"""

import argparse

import flax.serialization
import numpy as np
import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--check", default=None, help="crop shard (image uint8) for reference tokens")
  a = p.parse_args()
  net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
  sd = {k: v.detach().float().numpy() for k, v in net.state_dict().items()}
  with torch.no_grad():
    x = torch.zeros(1, 3, 224, 224)
    pos = net.interpolate_pos_encoding(net.patch_embed(x).new_zeros(1, 257, 384), 224, 224).float().numpy()
  P = {"patch": {"kernel": sd["patch_embed.proj.weight"].transpose(2, 3, 1, 0), "bias": sd["patch_embed.proj.bias"]},
       "cls": sd["cls_token"], "pos": pos, "norm": {"scale": sd["norm.weight"], "bias": sd["norm.bias"]}}
  dense = lambda k: {"kernel": sd[k + ".weight"].T, "bias": sd[k + ".bias"]}
  for i in range(len(net.blocks)):
    b = f"blocks.{i}."
    P[f"block_{i}"] = {"norm1": {"scale": sd[b + "norm1.weight"], "bias": sd[b + "norm1.bias"]}, "qkv": dense(b + "attn.qkv"),
                       "proj": dense(b + "attn.proj"), "ls1": sd[b + "ls1.gamma"],
                       "norm2": {"scale": sd[b + "norm2.weight"], "bias": sd[b + "norm2.bias"]}, "fc1": dense(b + "mlp.fc1"),
                       "fc2": dense(b + "mlp.fc2"), "ls2": sd[b + "ls2.gamma"]}
  open(a.out, "wb").write(flax.serialization.msgpack_serialize({"params": P}))
  print("wrote", a.out)
  if a.check:
    im = np.load(a.check)["image"][:16]
    x = torch.from_numpy(im).float().permute(0, 3, 1, 2) / 255
    m = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
      t = net.forward_features((x - m) / s)["x_norm_patchtokens"].numpy()
    np.savez(a.out.replace(".msgpack", "_check.npz"), image=im, tokens=t)
    print("reference tokens", t.shape)


if __name__ == "__main__":
  main()
