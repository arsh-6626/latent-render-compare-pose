"""Per-sample SYMSOL-T evaluation of a LiePose checkpoint (same sampler as Testbed.run_test).

Saves every sample's rotation error (min geodesic to the symmetry-equivalent set, deg) and
translation error, then prints mean (= the paper's spread), median and tail statistics per shape.

  DATADIR=... PYTHONPATH=src python scripts/eval_symsolt_samples.py <exp_path> --step 800000 --out errs.npz
"""

import argparse
import os

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from liepose.exp.symsolt import run as R
from liepose.utils.exploration import Options

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("exp")
  p.add_argument("--step", type=int, default=800000)
  p.add_argument("--n_slices", type=int, default=1, help="samples per image")
  p.add_argument("--steps", type=int, default=100, help="denoising steps")
  p.add_argument("--out", required=True)
  p.add_argument("--per_shape", type=int, default=None, help="evaluate only the first N test images of each shape")
  a_cli = p.parse_args()

  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(a_cli.exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(a_cli.exp, "ckpt"), a_cli.step
  tb = R.get_testbed(a.lie_type)(a, a_cli.exp, setup=True)
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=a_cli.steps, batch_size=100, n_slices=a_cli.n_slices)
  lm = tb.lie_metrics

  get_batch = tb.create_get_flat_batch_test_fn()
  backbone = jax.jit(tb.create_backbone_apply())
  p_sample = jax.jit(tb.create_p_sample_apply())
  T = tb.noise_schedule.timesteps
  cur = (np.linspace(T, 0, a_cli.steps, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))

  @jax.jit
  def errors(r, eq, t_gt):
    g = jax.vmap(lm.as_lie)(r)
    Rm = jax.vmap(lambda x: x.rotation().as_matrix())(g)
    Req = jax.vmap(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix()))(eq)
    tr = jnp.einsum("bij,bkij->bk", Rm, Req)
    ang = jnp.degrees(jnp.arccos(jnp.clip((tr - 1) / 2, -1, 1))).min(1)
    return ang, jnp.linalg.norm(jax.vmap(lambda x: x.translation())(g) - t_gt, axis=-1)

  @jax.jit
  def poses(r):
    g = jax.vmap(lm.as_lie)(r)
    return jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g)

  test = tb.create_test_dataset()
  if a_cli.per_shape is not None:
    lab = np.array([x["label_shape"] for x in test.dataset.annos])
    sel = np.concatenate([np.nonzero(lab == s)[0][: a_cli.per_shape] for s in range(len(SHAPES))])
    test.batches = np.array_split(sel, np.arange(0, len(sel), 100))[1:]

  key = jax.random.PRNGKey(a.seed + 2)
  rot_err, tran_err, labels, Rs, ts, idxs = [], [], [], [], [], []
  for i, data in enumerate(tqdm(test, desc="eval")):
    batch = get_batch(jax.random.PRNGKey(i), data, n_slices=a_cli.n_slices)
    f = backbone(batch["img"])
    rt = batch["rt"]
    for t, tp in seq:
      key, k = jax.random.split(key)
      _, rt = p_sample(k, f, rt, t, tp)
    s = a_cli.n_slices
    eq = jnp.repeat(jnp.asarray(data["rotations_equivalent"]), s, 0)
    t_gt = jnp.repeat(jnp.asarray(data["translation"]), s, 0)
    ang, te = errors(rt, eq, t_gt)
    rot_err.append(np.asarray(ang))
    tran_err.append(np.asarray(te))
    labels.append(np.repeat(np.asarray(data["label_shape"]), s))
    Rp, tp_ = poses(rt)
    Rs.append(np.asarray(Rp))
    ts.append(np.asarray(tp_))
    idxs.append(np.repeat(np.asarray(data["index"]), s))

  rot_err, tran_err, labels, Rs, ts, idxs = map(np.concatenate, (rot_err, tran_err, labels, Rs, ts, idxs))
  np.savez(a_cli.out, rot_err=rot_err, tran_err=tran_err, label=labels, R=Rs, t=ts, index=idxs, n_slices=a_cli.n_slices)
  print(f"{'shape':6s} {'R mean':>7s} {'R median':>8s} {'R p99':>7s} {'>10deg %':>8s} {'mean w/o >10':>12s} {'t mean':>7s} {'t median':>8s}")
  for s, name in enumerate(SHAPES + ["all"]):
    m = labels == s if name != "all" else np.ones_like(labels, bool)
    r, t = rot_err[m], tran_err[m]
    print(f"{name:6s} {r.mean():7.3f} {np.median(r):8.3f} {np.percentile(r, 99):7.2f} {100 * (r > 10).mean():8.2f} "
          f"{r[r <= 10].mean():12.3f} {t.mean():7.4f} {np.median(t):8.4f}")


if __name__ == "__main__":
  main()
