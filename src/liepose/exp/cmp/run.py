"""Train + evaluate a world-model-conditioned LiePose variant on SYMSOL-T SE(3).

  DATADIR=... python -m liepose.exp.cmp.run --root_path $LOGS/symsolt-cmp \
      "cond_mode=[compare]" "n_slices=[64]" "wm_dir=[$LOGS/probe/v3_soft15_full]" "cache_dir=[$DATADIR/dino_cache]"
"""

import json
import os
from typing import List

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from tqdm import tqdm

from liepose.exp.symsolt import run as symsolt_run
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Options

from .testbed import Testbed

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]


def build_explorations():
  e = symsolt_run.build_explorations()
  e.cond_mode = ["compare"]
  e.cond_gate_t = [-1]
  e.named_cond = [False]
  e.init_from = [None]  # checkpoint dir of a trained (vanilla) LiePose run to start from
  e.init_step = [None]
  e.den_mode = [None]  # None: ResNet + MLP head (cmp.Testbed); "dit" | "dit_nowm" | "energy": cmp.dit.Testbed
  e.dit_dim = [256]
  e.dit_depth = [4]
  e.fm_steps = [20]  # Euler steps for flow-matching sampling
  e.wm_kind = ["probe"]  # "probe": v3/v4 world model (dit.py); "ret": v7 crop + retrieval world model (dit_ret.py)
  e.wm_dir = [None]
  e.cache_dir = [None]
  e.cache_train = ["train-225k"]
  e.cache_test = ["test-5k"]
  e.n_slices = [64]
  e.ckpt_steps = [50000]
  e.skip_test = [True]  # the upstream run_test cannot pass the conditioning; evaluate() below replaces it
  e.eval_per_shape = [5000]
  e.eval_samples = [1]
  e.eval_tag = [""]  # suffix of the eval dir (e.g. "_k16" for a K-sample selection run)
  return e


def evaluate(tb: Testbed, out_dir, per_shape, n_samples, steps=100):
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=steps, batch_size=100, n_slices=n_samples)
  sample = tb.create_sampler(steps)
  lm, Lie = tb.lie_metrics, tb.Lie
  cache = tb.cache("test")
  test = tb.create_test_dataset()
  lab = np.array([x["label_shape"] for x in test.dataset.annos])
  sel = np.concatenate([np.nonzero(lab == s)[0][:per_shape] for s in range(len(SHAPES))])
  test.batches = np.array_split(sel, np.arange(0, len(sel), 100))[1:]
  prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), tb.a.repr_type)))

  @jax.jit
  def errors(rt, eq, t_gt):
    g = jax.vmap(lm.as_lie)(rt)
    R = jax.vmap(lambda x: x.rotation().as_matrix())(g)
    Req = jax.vmap(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix()))(eq)
    ang = jnp.degrees(jnp.arccos(jnp.clip((jnp.einsum("bij,bkij->bk", R, Req) - 1) / 2, -1, 1))).min(1)
    return ang, jnp.linalg.norm(jax.vmap(lambda x: x.translation())(g) - t_gt, axis=-1)

  @jax.jit
  def poses(rt):
    g = jax.vmap(lm.as_lie)(rt)
    return jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g)

  # the energy denoiser's own scalar energy at the last step: a free self-verification score for K samples
  self_energy = tb.create_energy_fn() if n_samples > 1 and hasattr(tb, "create_energy_fn") else None
  key = jax.random.PRNGKey(tb.a.seed + 2)
  rot, tran, labels, Rs, ts, idxs, Es = [], [], [], [], [], [], []
  for data in tqdm(test, desc="eval", leave=False):
    s = n_samples
    img = jnp.asarray(data["image"], jnp.float32) / 127.5 - 1.0
    tok, fg = cache.gather(data["index"], data["label_shape"])
    extra = tuple(jnp.asarray(x) for x in cache.extra(data["index"])) if hasattr(cache, "extra") else ()
    shape = jnp.asarray(data["label_shape"], jnp.int32)
    key, k1, k2 = jax.random.split(key, 3)
    rt0 = prior(jax.random.split(k1, len(img) * s))
    rt = sample(k2, img, shape, jnp.asarray(tok, jnp.float32), jnp.asarray(fg), rt0, *extra)
    a_, t_ = errors(rt, jnp.repeat(jnp.asarray(data["rotations_equivalent"]), s, 0),
                    jnp.repeat(jnp.asarray(data["translation"]), s, 0))
    rot.append(np.asarray(a_))
    tran.append(np.asarray(t_))
    labels.append(np.repeat(np.asarray(data["label_shape"]), s))
    if s > 1:
      Rp, tp = poses(rt)
      Rs.append(np.asarray(Rp))
      ts.append(np.asarray(tp))
      idxs.append(np.repeat(np.asarray(data["index"]), s))
      if self_energy is not None:
        Es.append(np.asarray(self_energy(rt, shape, jnp.asarray(tok, jnp.float32), jnp.asarray(fg), *extra)))
  rot, tran, labels = map(np.concatenate, (rot, tran, labels))
  os.makedirs(out_dir, exist_ok=True)
  extra = {}
  if n_samples > 1:  # per-sample poses, contiguous per image (verifier_eval.py format)
    extra = dict(R=np.concatenate(Rs), t=np.concatenate(ts), index=np.concatenate(idxs), n_slices=n_samples)
    if Es:
      extra["self_energy"] = np.concatenate(Es)
  np.savez(os.path.join(out_dir, "errors.npz"), rot_err=rot, tran_err=tran, label=labels, **extra)
  summary = {}
  for si, name in enumerate(SHAPES + ["all"]):
    m = labels == si if name != "all" else np.ones_like(labels, bool)
    r, t = rot[m], tran[m]
    summary[name] = {"rot_mean": float(r.mean()), "rot_median": float(np.median(r)), "rot_gt10_pct": float(100 * (r > 10).mean()),
                     "rot_lt2_pct": float(100 * (r < 2).mean()), "tran_mean": float(t.mean()), "tran_median": float(np.median(t))}
    print(name.ljust(6), {k: round(v, 4) for k, v in summary[name].items()}, flush=True)
  with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
  return summary


def main():
  a = get_args(exp_name="symsolt-cmp", desc="World-model-conditioned LiePose on SYMSOL-T")
  explorations: List[Options] = load_config_from_args(a, build_explorations()).expand()
  for opt in explorations:
    opt.pop("sample_options", None)
    exp_name = opt.to_string(includes=["den_mode", "cond_mode", "cond_gate_t", "n_slices", "train_steps", "seed"])
    exp_path = os.path.join(a.root_path, exp_name)
    if opt.get("den_mode") and opt.get("wm_kind") == "ret":
      from .dit_ret import Testbed as DiTTestbed
      tb = DiTTestbed(opt, exp_path, setup=True)
    elif opt.get("den_mode"):
      from .dit import Testbed as DiTTestbed
      tb = DiTTestbed(opt, exp_path, setup=True)
    else:
      tb = Testbed(opt, exp_path, setup=True)
    if not opt.skip_train:
      save_config(opt, os.path.join(exp_path, "configs.yaml"))
      tb.run_train()
    evaluate(tb, os.path.join(exp_path, f"eval_{int(tb.state.step)}{opt.get('eval_tag', '')}"), opt.eval_per_shape,
             opt.eval_samples)


if __name__ == "__main__":
  main()
