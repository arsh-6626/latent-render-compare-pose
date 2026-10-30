"""EMA vs live weights of a field-denoiser checkpoint (sp_field_dit / tless_field_dit), by the training loss on fresh training
batches (supplement, tab:supp-ema): both weight sets are scored on the same batches (a data / noise seed the run never
used), and the lower mean loss picks the weights used for testing. No test data is read.

  python scripts/speedplus/ema_check.py --module sp_field_dit --den_mode field_full_lat --field_dir <probe>/sp_field_v1 \
      --restore <exp>/ckpt --step 400000 --out <dir>/ema_check.json
"""

import argparse
import importlib
import json
import os
import sys

import jax
import numpy as np

from liepose.utils.config import get_args, load_config_from_args


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--module", required=True, choices=["sp_field_dit", "tless_field_dit"])
  p.add_argument("--den_mode", required=True)
  p.add_argument("--field_dir", required=True)
  p.add_argument("--restore", required=True)
  p.add_argument("--step", type=int, required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--n_batches", type=int, default=50)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--n_slices", type=int, default=16)
  p.add_argument("--seed", type=int, default=777, help="data / noise seed for the fresh batches (training runs use 42)")
  a = p.parse_args()
  mod = importlib.import_module(f"liepose.exp.cmp.{a.module}")
  tb_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "_tb_ema")
  sys.argv = [sys.argv[0], "--root_path", tb_dir, f"den_mode=[{a.den_mode}]", f"field_dir=[{a.field_dir}]", "train_steps=[100000]",
              f"batch_size=[{a.batch_size}]", f"n_slices=[{a.n_slices}]", "grad_accum=[1]", f"seed=[{a.seed}]", "skip_train=[True]",
              ] + ([f"restore_path=[{a.restore}]", f"restore_step=[{a.step}]"] if a.restore != "none" else [])  # none: init (smoke test)
  ga = get_args(exp_name="ema_check", desc="EMA vs live training loss")
  opt = next(iter(load_config_from_args(ga, mod.build_explorations()).expand()))
  opt.pop("sample_options", None)
  tb = mod.Testbed(opt, tb_dir, setup=True)
  assert int(tb.state.step) in (a.step, a.step + 1), int(tb.state.step)
  data = iter(tb.create_train_dataset())
  flat = tb.create_get_flat_batch_train_fn()
  upd = jax.jit(tb.create_update_fn())  # returns the loss at the given params (before its update); the update is discarded
  st_live, st_ema = tb.state, tb.state.replace(params=tb.state.params_ema)
  rng = np.random.default_rng(a.seed)
  loss = {"ema": [], "live": []}
  for i in range(a.n_batches):
    b = flat(jax.random.PRNGKey(int(rng.integers(2**31))), next(data), n_slices=a.n_slices)
    loss["live"].append(float(upd(st_live, b)[0]))
    loss["ema"].append(float(upd(st_ema, b)[0]))
    if i % 10 == 0:
      print(f"  {i}/{a.n_batches} live {np.mean(loss['live']):.4f} ema {np.mean(loss['ema']):.4f}", flush=True)
  d = np.array(loss["ema"]) - np.array(loss["live"])
  rel = jax.tree_util.tree_map(lambda x, y: (float(np.sum((np.asarray(x) - np.asarray(y)) ** 2)), float(np.sum(np.asarray(y) ** 2))),
                               tb.state.params_ema, tb.state.params)
  num, den = map(sum, zip(*jax.tree_util.tree_leaves(rel, is_leaf=lambda x: isinstance(x, tuple))))
  res = {"restore": a.restore, "step": a.step, "den_mode": a.den_mode, "n_batches": a.n_batches, "batch_size": a.batch_size,
         "n_slices": a.n_slices, "seed": a.seed, "loss_ema": float(np.mean(loss["ema"])), "loss_live": float(np.mean(loss["live"])),
         "diff_mean": float(d.mean()), "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))), "rel_param_dist": float(np.sqrt(num / den)),
         "use": "ema" if np.mean(loss["ema"]) <= np.mean(loss["live"]) else "live"}
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  json.dump(res, open(a.out, "w"), indent=1)
  print(json.dumps(res), flush=True)


if __name__ == "__main__":
  main()
