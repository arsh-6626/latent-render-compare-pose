"""Draw K LiePose samples on exported T-LESS *training* crops (for training the joint scorer / checking the
refiner on realistic hypotheses). Output matches verifier_eval's *_samples.npz, plus `row` = index into the
train token cache (tless_tokens/train_*).

  DATADIR=... PYTHONPATH=src python scripts/tless/sample_train.py <tless_exp> --step 400000 --no_ema \
      --crops $DATADIR/tless_crops/train --tokens $DATADIR/tless_tokens --n 8000 --k 16 --out train_samples.npz
"""

import argparse
import glob
import os

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import liepose.exp.bop.dataset_registry  # noqa: F401
from liepose.exp.bop import run as bop_run
from liepose.exp.jepa_probe.probe_tless import load_symmetries
from liepose.utils.exploration import Options


def main():
  p = argparse.ArgumentParser()
  p.add_argument("exp")
  p.add_argument("--step", type=int, default=400000)
  p.add_argument("--no_ema", action="store_true")
  p.add_argument("--crops", required=True)
  p.add_argument("--tokens", required=True)
  p.add_argument("--n", type=int, default=8000)
  p.add_argument("--k", type=int, default=16)
  p.add_argument("--batch", type=int, default=32)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", required=True)
  c = p.parse_args()

  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(c.exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(c.exp, "ckpt"), c.step
  if c.no_ema:
    a.ema_enabled = False
  tb = bop_run.get_testbed("se3")(a, c.exp, setup=True)
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=c.batch, n_slices=c.k)
  lm, ns = tb.lie_metrics, tb.noise_schedule
  get_batch = tb.create_get_flat_batch_test_fn()
  backbone = jax.jit(tb.create_backbone_apply())
  p_sample = jax.jit(tb.create_p_sample_apply())
  T = ns.timesteps
  cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))

  @jax.jit
  def to_rt(r):
    g = jax.vmap(lm.as_lie)(r)
    return jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g)

  syms = load_symmetries(os.path.join(os.environ["DATADIR"], "bop_datasets/tless/models_eval/models_info.json"))
  # the token cache was built from the shards in sorted order, so crop j of the concatenated shards is token row j
  shards = sorted(glob.glob(os.path.join(c.crops, "shard_*.npz")))
  n_cache = len(np.load(os.path.join(c.tokens, "train_meta.npz"))["object_id"])
  sizes = [len(np.load(s, mmap_mode="r")["object_id"]) for s in shards]
  offsets = np.cumsum([0] + sizes)
  rows = np.sort(np.random.default_rng(c.seed).choice(n_cache, c.n, replace=False))

  out = {k: [] for k in ["R", "t", "rot_err", "tran_err", "row", "R_gt", "t_gt", "obj"]}
  key = jax.random.PRNGKey(c.seed + 5)
  for si, s in enumerate(shards):
    sel = rows[(rows >= offsets[si]) & (rows < offsets[si + 1])]
    if len(sel) == 0 or offsets[si] >= n_cache:
      continue
    d = np.load(s)
    loc = sel - offsets[si]
    for b0 in tqdm(range(0, len(loc), c.batch), desc=os.path.basename(s), leave=False):
      li = loc[b0 : b0 + c.batch]
      data = {"image": d["image"][li].astype(np.float32), "mask": d["mask"][li][..., None].astype(np.float32)}
      batch = get_batch(jax.random.PRNGKey(si * 10000 + b0), data, n_slices=c.k)
      feat = backbone(batch["img"])
      rt = batch["rt"]
      for t, tp in seq:
        key, k1 = jax.random.split(key)
        _, rt = p_sample(k1, feat, rt, t, tp)
      R, tr_ = map(np.asarray, to_rt(rt))
      B = len(li)
      R, tr_ = R.reshape(B, c.k, 3, 3), tr_.reshape(B, c.k, 3)
      for j, l in enumerate(li):
        oid = int(d["object_id"][l])
        R_gt = np.asarray(jaxlie.SO3(jnp.asarray(d["rotation_allo"][l])).as_matrix())
        t_gt = d["translation_scaled"][l]
        eq = R_gt @ syms[oid]
        tr = np.einsum("kij,mij->km", R[j], eq)
        out["rot_err"].append(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min(1))
        out["tran_err"].append(np.linalg.norm(tr_[j] - t_gt, axis=-1))
        for k_, v_ in [("R", R[j]), ("t", tr_[j]), ("row", offsets[si] + l), ("R_gt", R_gt), ("t_gt", t_gt), ("obj", oid)]:
          out[k_].append(v_)
  out = {k: np.stack(v) for k, v in out.items()}
  np.savez(c.out, k=c.k, **out)
  r = out["rot_err"]
  print(f"train crops {len(r)}  per-sample R median {np.median(r):.2f}  >30deg {100 * (r > 30).mean():.2f}%  "
        f"crops with >=1 sample >30deg {100 * (r > 30).any(1).mean():.1f}%")


if __name__ == "__main__":
  main()
