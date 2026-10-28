"""N samples per test crop for an sp_dit checkpoint (the same sampler as sp_dit.test, full splits), saved as
<split>_samples.npz for consensus_score.py: index (n,), R (n, S, 3, 3) egocentric, t (n, S, 3) camera frame.
Inference reads only the test crop (tokens / images) and its box meta; GT is read by consensus_score.py only.

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit:scripts/speedplus python scripts/speedplus/test_consensus.py \
      --den_mode dit_ft --wm_dir $LOGS/probe/sp_geo_v1 --restore $LOGS/ft400_ckpt/dit_ft --step 100000 --out <dir>
"""

import argparse
import glob
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esa_score import allo_to_ego  # noqa: E402

from liepose.data.speedplus.dataset import recover_translation  # noqa: E402
from liepose.exp.cmp import sp_dit  # noqa: E402
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR  # noqa: E402
from liepose.exp.jepa_probe.probe_tless import TokenSplit  # noqa: E402
from liepose.exp.jepa_probe.sp_geometry import crop_meta  # noqa: E402
from liepose.utils.config import get_args, load_config_from_args  # noqa: E402
from liepose.utils.exploration import Options  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--den_mode", required=True)
  p.add_argument("--wm_dir", required=True)
  p.add_argument("--restore", required=True)
  p.add_argument("--step", type=int, required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--splits", nargs="+", default=["lightbox", "sunlamp"])
  p.add_argument("--n_samples", type=int, default=16)
  p.add_argument("--crops_per_batch", type=int, default=25)
  p.add_argument("--limit", type=int, default=0, help="smoke tests only (consensus_score.py rejects partial splits)")
  a = p.parse_args()
  sys.argv = [sys.argv[0], "--root_path", os.path.join(a.out, "_tb"), f"den_mode=[{a.den_mode}]", f"wm_dir=[{a.wm_dir}]",
              "train_steps=[400000]", "batch_size=[64]", "n_slices=[64]", "skip_train=[True]", f"restore_path=[{a.restore}]",
              f"restore_step=[{a.step}]"]
  ga = get_args(exp_name="consensus", desc="consensus")
  opt = next(iter(load_config_from_args(ga, sp_dit.build_explorations()).expand()))
  opt.pop("sample_options", None)
  tb = sp_dit.Testbed(opt, os.path.join(a.out, "_tb"), setup=True)
  assert int(tb.state.step) in (a.step, a.step + 1), int(tb.state.step)
  tb.opt = Options(dict(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=100, n_slices=1))
  D = os.environ["DATADIR"]
  lm, Lie = tb.lie_metrics, tb.Lie
  prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), tb.a.repr_type)))
  sample, poses = tb.create_sampler(100), tb._poses_fn()
  S = a.n_samples
  os.makedirs(a.out, exist_ok=True)
  for split in a.splits:
    obs = TokenSplit(os.path.join(D, "speedplus_tokens"), split)
    meta, idx, _ = crop_meta(os.path.join(D, "speedplus_crops", split))
    assert (idx == obs.index).all()
    sh = sorted(glob.glob(os.path.join(D, "speedplus_crops", split, "shard_*.npz")))
    ims = np.concatenate([np.load(f)["image"] for f in sh]) if tb.ft else None
    box = {k: np.concatenate([np.load(f)[k] for f in sh]) for k in ["bbox", "bbox_center", "focal_ratio"]}
    key = jax.random.PRNGKey(tb.a.seed + 5)
    Rall, tall = [], []
    n = len(obs.index) if not a.limit else a.limit
    for s in range(0, n, a.crops_per_batch):
      rows = np.repeat(np.arange(s, min(s + a.crops_per_batch, n)), S)
      key, k1, k2 = jax.random.split(key, 3)
      rt = sample(k2, prior(jax.random.split(k1, len(rows))), None if tb.ft else jnp.asarray(obs.tokens[rows], jnp.float32),
                  jnp.asarray(FG_FLOOR + obs.fg[rows]), jnp.asarray(meta[rows]), None if ims is None else jnp.asarray(ims[rows], jnp.float32))
      Rg, tg = poses(rt)
      Rall.append(np.asarray(Rg, np.float64))
      tall.append(np.asarray(tg, np.float64))
      if (s // a.crops_per_batch) % 20 == 0:
        print(f"  {split} {min(s + a.crops_per_batch, n)}/{n}", flush=True)
    R_allo, t_sc = np.concatenate(Rall), np.concatenate(tall)
    rep = lambda x: np.repeat(x[:n], S, 0)
    t_est = recover_translation(t_sc, rep(box["bbox"]), rep(box["bbox_center"]), rep(box["focal_ratio"]), tb.a.tran_scale, tb.a.tran_offset)
    R_est = allo_to_ego(R_allo, t_est)
    np.savez(os.path.join(a.out, f"{split}_samples.npz"), index=obs.index[:n].astype(np.int64), R=R_est.reshape(n, S, 3, 3), t=t_est.reshape(n, S, 3))
    print(f"{split}: saved {n} x {S} samples", flush=True)


if __name__ == "__main__":
  main()
