"""Screen of candidate comparison spaces for the world model (rebuttal of "DINOv2 is too invariant"): how much each
feature space changes under small, known pose changes of the object, relative to how much it differs between crops.

Images: --n masked T-LESS training crops (tless_crops/train, image x visible mask, as the token caches), each with
perturbations that are exact pose changes for a crop: in-plane rotation about the crop centre (= camera roll),
zoom about the centre (~ depth change of the same factor), shift (~ image-plane translation).

Per extractor (tokens on its own grid, pooled to <= 8 x 8; distance = mean squared token difference over the tokens
covered by the object in the unperturbed crop):
  S(delta) = median over crops of d(x, T_delta x) / median over same-object crop pairs of d(x, x')
A larger S at small delta = more sensitive (less invariant) to small pose changes; a field fitted to that space has
more signal above its fitting error. No training, no test images, no ground truth beyond the crop's own mask.

  DATADIR=... PYTHONPATH=src python scripts/screen/feature_sensitivity.py --out results/screen/sens.json \
      --liepose_ckpt $LOGS/experiments/tless-se3-hpc/ckpt/checkpoint_400000 [--ft_ckpt <field *_ft ckpt dir>]
"""

import argparse
import glob
import json
import os

import flax
import jax
import jax.numpy as jnp
import jax_resnet
import numpy as np
import orbax.checkpoint as ocp
from scipy import ndimage

from liepose.models.dinov2 import MEAN, STD, Block

ROT = [0.5, 1.0, 2.0, 5.0, 10.0]  # deg
ZOOM = [1.01, 1.02, 1.05, 1.10]
SHIFT = [1, 2, 4, 8]  # px


def perturb(img, kind, v):
  """img (H, W, 3) uint8 masked crop -> perturbed copy (black background, as the masked crops)"""
  H, W = img.shape[:2]
  c = np.array([(H - 1) / 2, (W - 1) / 2])
  if kind == "rot":
    a = np.deg2rad(v)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
  elif kind == "zoom":
    M = np.eye(2) / v
  else:
    M = np.eye(2)
  off = c - M @ c - (np.array([0, v]) if kind == "shift" else 0)
  out = np.stack([ndimage.affine_transform(img[..., k].astype(np.float32), M, offset=off, order=1, mode="constant")
                  for k in range(3)], -1)
  return np.clip(out, 0, 255).astype(np.uint8)


class DinoLayers(flax.linen.Module):
  """liepose.models.dinov2.DinoV2 with the patch tokens after the requested blocks (last: + final LayerNorm)"""
  layers: tuple = (4, 6, 8, 12)

  @flax.linen.compact
  def __call__(self, image):
    dim, grid = 384, 16
    x = (image / 255.0 - MEAN) / STD
    x = flax.linen.Conv(dim, (14, 14), strides=14, padding="VALID", name="patch")(x).reshape(image.shape[0], -1, dim)
    cls = jnp.broadcast_to(self.param("cls", flax.linen.initializers.zeros, (1, 1, dim)), (x.shape[0], 1, dim))
    x = jnp.concatenate([cls, x], 1) + self.param("pos", flax.linen.initializers.zeros, (1, 1 + grid ** 2, dim))
    out = {}
    for i in range(12):
      x = Block(dim, 6, name=f"block_{i}")(x)
      if i + 1 in self.layers and i + 1 != 12:
        out[i + 1] = x[:, 1:]
    out[12] = flax.linen.LayerNorm(epsilon=1e-6, name="norm")(x)[:, 1:]
    return out


def pool_to(f, g):
  """(N, h, w, C) -> (N, g*g, C) by average pooling (h divisible by g) or as is when h <= g"""
  N, h, w, C = f.shape
  if h <= g:
    return f.reshape(N, h * w, C)
  k = h // g
  return f[:, :g * k, :g * k].reshape(N, g, k, g, k, C).mean((2, 4)).reshape(N, g * g, C)


def mask_tokens(mask, g):
  """(N, 224, 224) -> (N, g*g) object coverage per token"""
  N = mask.shape[0]
  k = 224 // g
  return mask[:, :g * k, :g * k].reshape(N, g, k, g, k).mean((2, 4)).reshape(N, g * g)


def restore_np(path):
  ck = ocp.PyTreeCheckpointer()
  tree = ck.metadata(path).item_metadata
  tree = getattr(tree, "tree", tree)
  return ck.restore(path, restore_args=jax.tree_util.tree_map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), tree))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--n", type=int, default=1024)
  p.add_argument("--out", required=True)
  p.add_argument("--liepose_ckpt", default=None)
  p.add_argument("--ft_ckpt", default=None, help="a *_ft field-denoiser checkpoint: its trained DINOv2 (params/dino)")
  p.add_argument("--ft_key", default="dino", help="dino (trained copy) or dino_frozen (joint runs: the EMA comparison encoder)")
  p.add_argument("--only_ft", action="store_true", help="skip the other extractors")
  p.add_argument("--bs", type=int, default=64)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  z = np.load(sorted(glob.glob(os.path.join(D, "tless_crops/train/shard_*.npz")))[0])
  rng = np.random.default_rng(0)
  sel = np.sort(rng.choice(len(z["index"]), a.n, replace=False))
  img = (z["image"][sel].astype(np.float32) * z["mask"][sel][..., None]).astype(np.uint8)
  mask, obj = z["mask"][sel].astype(np.float32), z["object_id"][sel]
  kinds = [("orig", 0)] + [("rot", v) for v in ROT] + [("zoom", v) for v in ZOOM] + [("shift", v) for v in SHIFT]
  print("perturbing", flush=True)
  ims = {k: (img if k[0] == "orig" else np.stack([perturb(x, *k) for x in img])) for k in kinds}
  # same-object pairs (different crops) for the normaliser
  pairs = []
  for o in np.unique(obj):
    i = np.where(obj == o)[0]
    if len(i) > 1:
      j = rng.permutation(i)
      pairs += [(x, y) for x, y in zip(i, j) if x != y]
  pairs = np.array(pairs)

  extractors = {}
  dparams = flax.serialization.msgpack_restore(open(os.path.join(D, "dinov2_vits14_224.msgpack"), "rb").read())
  dl = DinoLayers()
  fd = jax.jit(lambda x: dl.apply({"params": dparams["params"]}, x))
  extractors["dinov2"] = lambda x: {f"dinov2_L{k}": v.reshape(len(x), 16, 16, -1) for k, v in fd(x).items()}
  if a.ft_ckpt:
    ft = restore_np(a.ft_ckpt)
    fp = ft.get("params_ema") or ft["params"]
    fp = fp.get("params", fp)
    ftd = jax.jit(lambda x: dl.apply({"params": fp[a.ft_key]}, x))
    extractors["dinov2_ft"] = lambda x: {f"dinov2ft_L{k}": v.reshape(len(x), 16, 16, -1) for k, v in ftd(x).items() if k in (8, 12)}
  if a.only_ft:
    extractors = {k: v for k, v in extractors.items() if k in ("dinov2_ft",)} | {"dinov2": extractors["dinov2"]}
  RN, rparams = jax_resnet.pretrained_resnet(34)
  for name, var, norm in ([] if a.only_ft else [("r34_imagenet", rparams, "imagenet")]) + \
                         ([("r34_liepose", None, "liepose")] if a.liepose_ckpt and not a.only_ft else []):
    if var is None:
      ck = restore_np(a.liepose_ckpt)
      var = {"params": ck["params"]["backbone"]["resnet"], "batch_stats": ck["batch_stats"]["backbone"]["resnet"]}
    for end, tag in [(15, "l3"), (18, "l4")]:  # layers_0 stem, 1 pool, 2-4 / 5-8 / 9-14 / 15-17 the four stages
      m, v = jax_resnet.slice_model_and_variables(RN(), var if name == "r34_liepose" else rparams, start=0, end=end)
      f = jax.jit(lambda x, m=m, v=v: m.apply(v, x, mutable=False))

      def run(x, f=f, tag=tag, name=name, norm=norm):
        x = jnp.asarray(x, jnp.float32)
        x = (x / 255.0 - MEAN) / STD if norm == "imagenet" else x / 127.5 - 1.0
        o = f(x)
        return {f"{name}_{tag}": o[0] if isinstance(o, tuple) else o}
      extractors[f"{name}_{tag}"] = run

  feats = {}
  for ename, fn in extractors.items():
    print("extract", ename, flush=True)
    for k in kinds:
      outs = []
      for s in range(0, a.n, a.bs):
        o = fn(jnp.asarray(ims[k][s:s + a.bs], jnp.float32))
        outs.append({kk: np.asarray(pool_to(np.asarray(vv, np.float32), 8)) for kk, vv in o.items()})
      for kk in outs[0]:
        feats.setdefault(kk, {})[k] = np.concatenate([o[kk] for o in outs])

  res = {"n": a.n, "n_pairs": int(len(pairs)), "spaces": {}}
  for space, F in feats.items():
    g = int(round(np.sqrt(F[("orig", 0)].shape[1])))
    w = (mask_tokens(mask, g) > 0.1).astype(np.float32)
    w = np.maximum(w, 1e-6)
    d = lambda A, B, W: ((A - B) ** 2).mean(-1).__mul__(W).sum(-1) / W.sum(-1)
    F0 = F[("orig", 0)]
    between = float(np.median(d(F0[pairs[:, 0]], F0[pairs[:, 1]], np.maximum(w[pairs[:, 0]], w[pairs[:, 1]]))))
    r = {"grid": g, "between": between}
    for k in kinds[1:]:
      r[f"{k[0]}_{k[1]}"] = float(np.median(d(F0, F[k], w)) / between)
    res["spaces"][space] = r
    print(space, json.dumps({kk: round(vv, 4) for kk, vv in r.items()}), flush=True)
  os.makedirs(os.path.dirname(a.out), exist_ok=True)
  json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
