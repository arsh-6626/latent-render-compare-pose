"""Oracle check in the liepose (JAX) env: raw pixels, a freshly initialised target encoder, and
the probe checkpoint's EMA target encoder.

  PYTHONPATH=src:scripts/oracle python scripts/oracle/feats_jax.py --bank $DATADIR/oracle/bank.npz \
      --ckpt $DATADIR/logs/probe/nce1.0 --out $DATADIR/oracle/results_jax.json
"""

import argparse
import json
import os

import flax
import jax
import jax.numpy as jnp
import numpy as np
from metrics import print_table, score

from liepose.exp.jepa_probe.probe import POSE_DIM, PoseWorldModel


def distances(feats, n_bank):
  """feats: (S, N+Q, F) -> (S, Q, N) mean squared distance."""
  @jax.jit
  def one(f):
    b, q = f[:n_bank], f[n_bank:]
    d = (q**2).sum(-1)[:, None] + (b**2).sum(-1)[None] - 2 * q @ b.T
    return d / f.shape[-1]
  return np.stack([np.asarray(one(jnp.asarray(f))) for f in feats])


def encode_all(model, params, images, bs=128):
  enc = jax.jit(lambda x: model.apply(params, x, method=PoseWorldModel.encode))
  out = []
  for s in range(images.shape[0]):
    fs = [np.asarray(enc(images[s, i : i + bs])).reshape(len(images[s, i : i + bs]), -1)
          for i in range(0, images.shape[1], bs)]
    out.append(np.concatenate(fs).astype(np.float32))
  return np.stack(out)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--bank", required=True)
  p.add_argument("--ckpt", default=None, help="probe output dir containing params.msgpack and args.json")
  p.add_argument("--out", required=True)
  a = p.parse_args()

  bank = dict(np.load(a.bank))
  images = bank["images"]  # (S, N+Q, H, W, 3)
  n_bank = len(bank["R_bank"])
  results = {}

  # pixels, 56x56 grayscale: the pixel render-and-compare reference
  g = images.reshape(*images.shape[:2], 56, 4, 56, 4, 3).mean(axis=(3, 5, 6)).astype(np.float32) / 255
  results["pixels_56"] = score(distances(g.reshape(*g.shape[:2], -1), n_bank), bank)
  print_table("pixels_56", results["pixels_56"])

  cfg = {"dim": 384, "depth": 6, "pred_depth": 4}
  if a.ckpt:
    with open(os.path.join(a.ckpt, "args.json")) as f:
      args = json.load(f)
    cfg = {k: args[k] for k in cfg}
    cfg["predictor_type"] = args.get("predictor", "additive")
  model = PoseWorldModel(heads=cfg["dim"] // 64, **cfg)
  x0 = jnp.zeros((1, 224, 224, 3), jnp.uint8)
  init = model.init(jax.random.PRNGKey(0), x0, jnp.zeros((1, POSE_DIM)), x0)

  results["random_init"] = score(distances(encode_all(model, init, images), n_bank), bank)
  print_table("random_init", results["random_init"])

  if a.ckpt:
    with open(os.path.join(a.ckpt, "params.msgpack"), "rb") as f:
      state = flax.serialization.from_bytes({"params": init, "ema_params": init}, f.read())
    for key in ["ema_params", "params"]:
      name = f"probe_{key}"
      results[name] = score(distances(encode_all(model, state[key], images), n_bank), bank)
      print_table(name, results[name])

  with open(a.out, "w") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  main()
