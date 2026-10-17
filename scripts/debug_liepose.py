"""Localise why the LiePose SYMSOL-T checkpoint samples as random.

On one training batch, compare the denoising loss computed three ways:
  (i)   training path: state.apply_fn with mutable batch_stats (BatchNorm in train mode)
  (ii)  sampler path: create_backbone_apply + create_head_apply (EMA / running stats)
  (ii') sampler path without EMA
  (iii) sampler path with image features shuffled across the batch

  DATADIR=... PYTHONPATH=src python scripts/debug_liepose.py <exp_path> [step]
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from liepose.exp.symsolt import run as R
from liepose.utils.exploration import Options


def main():
  exp = sys.argv[1]
  step = int(sys.argv[2]) if len(sys.argv) > 2 else 800000
  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path = os.path.join(exp, "ckpt")
  a.restore_step = step
  tb = R.get_testbed(a.lie_type)(a, exp, setup=True)
  print("restored step", int(tb.state.step))

  ds = iter(tb.create_train_dataset())
  get_batch = tb.create_get_flat_batch_train_fn()
  batch = get_batch(jax.random.PRNGKey(0), next(ds), n_slices=32)
  dist = jax.vmap(tb.lie_metrics.get_distance_fn(tb.a.loss_name))
  target = batch["zt"] if tb.a.learn_noise else batch["r0"]
  t = batch["t"].reshape((-1, 1))
  print("batch", {k: v.shape for k, v in batch.items()})

  st = tb.state
  mu_train, _ = st.apply_fn({"params": st.params, "batch_stats": st.batch_stats},
                            batch["img"], batch["rt"], t, mutable=["batch_stats"])
  print("(i)   train path          loss", float(dist(target, mu_train).mean()))

  for ema in [True, False]:
    tb.a.ema_enabled = ema
    feat = jax.jit(tb.create_backbone_apply())(batch["img"])
    head = jax.jit(tb.create_head_apply())
    mu = head(feat, batch["rt"], t)
    print(f"(ii)  sampler path ema={ema!s:5s} loss", float(dist(target, mu).mean()))
    perm = np.random.default_rng(0).permutation(feat.shape[0])
    mu_s = head(feat[perm], batch["rt"], t)
    print(f"(iii) shuffled feat ema={ema!s:5s} loss", float(dist(target, mu_s).mean()))

  # unconditional reference: predict zero noise
  print("      zero prediction     loss", float(dist(target, jnp.zeros_like(mu)).mean()))
  # BatchNorm running statistics sanity
  bs = jax.tree_util.tree_leaves(st.batch_stats)
  print("batch_stats leaves", len(bs), "mean |x|", float(np.mean([np.abs(np.asarray(x)).mean() for x in bs])))


if __name__ == "__main__" and len(sys.argv) == 2:
  main()


def trace(exp, step=800000, n_slices=8):
  """Run the testbed's own reverse sampler on a training batch and log error per step."""
  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(exp, "ckpt"), step
  tb = R.get_testbed(a.lie_type)(a, exp, setup=True)
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=16, n_slices=n_slices)
  lm = tb.lie_metrics

  data = next(iter(tb.create_train_dataset()))
  gb = tb.create_get_flat_batch_test_fn()
  batch = gb(jax.random.PRNGKey(3), {k: v for k, v in data.items()}, n_slices=n_slices)
  print("test-batch keys", {k: getattr(v, "shape", None) for k, v in batch.items()})
  R_gt = np.repeat(np.asarray(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix())(jnp.asarray(data["rotation"]))), n_slices, 0)
  t_gt = np.repeat(np.asarray(data["translation"]), n_slices, 0)

  def err(r):
    g = jax.vmap(lm.as_lie)(r)
    Rm = np.asarray(jax.vmap(lambda x: x.rotation().as_matrix())(g))
    tr = np.einsum("bij,bij->b", Rm, R_gt)
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    tt = np.asarray(jax.vmap(lambda x: x.translation())(g))
    return np.median(ang), np.median(np.linalg.norm(tt - t_gt, axis=-1))

  feat = jax.jit(tb.create_backbone_apply())(batch["img"])
  p = jax.jit(tb.create_p_sample_apply())
  T = tb.noise_schedule.timesteps
  cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))
  rt = batch["rt"]
  print("init  rt err (deg, trans)", err(rt))
  key = jax.random.PRNGKey(0)
  for i, (t, tp) in enumerate(seq):
    key, k = jax.random.split(key)
    r0, rt = p(k, feat, rt, t, tp)
    if i % 10 == 0 or i == len(seq) - 1:
      print(f"t={t:3d} r0 err {err(r0)}  rt err {err(rt)}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "trace":
  import jaxlie
  trace(sys.argv[1])


def oracle(exp, n=64):
  """Reverse-sample with the TRUE noise (oracle mu) and with the model, both from GT-conditioned noise."""
  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(exp, "ckpt"), int(os.environ.get("STEP", 800000))
  tb = R.get_testbed(a.lie_type)(a, exp, setup=True)
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=16, n_slices=4)
  lm, Lie, ns = tb.lie_metrics, tb.Lie, tb.noise_schedule
  step = jax.jit(jax.vmap(tb.create_p_sample_fn()))

  data = next(iter(tb.create_train_dataset()))
  rot = jnp.repeat(jnp.asarray(data["rotation"]), 4, 0)
  tran = jnp.repeat(jnp.asarray(data["translation"]), 4, 0)
  r0_true = jax.vmap(lambda q, t: Lie.from_rotation_and_translation(jaxlie.SO3(q), t))(rot, tran)
  B = rot.shape[0]

  def err(r):
    g = jax.vmap(lm.as_lie)(r)
    d = jax.vmap(lambda x, y: (x.inverse() @ y).log())(r0_true, g)
    return float(jnp.median(jnp.linalg.norm(d[:, :3], axis=-1))), float(jnp.median(jnp.linalg.norm(d[:, 3:], axis=-1)))

  T = ns.timesteps
  cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))
  key = jax.random.PRNGKey(0)
  z = jax.vmap(lambda k: tb.LieDist._sample_unit(k, n=()))(jax.random.split(key, B))
  rt = jax.vmap(lambda r, zz: lm.as_repr(r @ Lie.exp(ns.sqrt_alphas[cur[0]] * zz), tb.a.repr_type))(r0_true, z)
  print("init err (tangent |v|, |w|)", err(rt))
  for i, (t, tp) in enumerate(seq):
    key, k = jax.random.split(key)
    sa = ns.sqrt_alphas[t]
    mu = jax.vmap(lambda r0, r: lm.as_repr((r0.inverse() @ lm.as_lie(r)).log() / sa, tb.a.repr_type))(r0_true, rt)
    r0e, rt = step(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
    if i % 20 == 0 or i == len(seq) - 1:
      print(f"oracle t={t:3d} r0 err {err(r0e)} rt err {err(rt)}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "oracle":
  import jaxlie
  oracle(sys.argv[1])


def model_check(exp, reps=4):
  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(exp, "ckpt"), int(os.environ.get("STEP", 800000))
  tb = R.get_testbed(a.lie_type)(a, exp, setup=True)
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=16, n_slices=reps)
  lm, Lie, ns = tb.lie_metrics, tb.Lie, tb.noise_schedule
  head = jax.jit(tb.create_head_apply())
  feat = jax.jit(tb.create_backbone_apply())
  step = jax.jit(jax.vmap(tb.create_p_sample_fn()))

  data = next(iter(tb.create_train_dataset()))
  img = jnp.asarray(data["image"], jnp.float32) / 127.5 - 1.0
  f = feat(img)
  rot = jnp.repeat(jnp.asarray(data["rotation"]), reps, 0)
  tran = jnp.repeat(jnp.asarray(data["translation"]), reps, 0)
  r0_true = jax.vmap(lambda q, t: Lie.from_rotation_and_translation(jaxlie.SO3(q), t))(rot, tran)
  B = rot.shape[0]

  def err(r):
    g = jax.vmap(lm.as_lie)(r)
    d = jax.vmap(lambda x, y: (x.inverse() @ y).log())(r0_true, g)
    return round(float(jnp.median(jnp.linalg.norm(d[:, :3], axis=-1))), 3), round(float(jnp.median(jnp.linalg.norm(d[:, 3:], axis=-1))), 3)

  key = jax.random.PRNGKey(0)
  z = jax.vmap(lambda k: tb.LieDist._sample_unit(k, n=()))(jax.random.split(key, B))
  print("one-shot r0 estimate from correctly-noised rt (|v| trans-part, |w| rot-part of log error):")
  for t in [99, 80, 60, 40, 20, 10, 5, 1]:
    sa = ns.sqrt_alphas[t]
    rt = jax.vmap(lambda r, zz: lm.as_repr(r @ Lie.exp(sa * zz), tb.a.repr_type))(r0_true, z)
    mu = head(f, rt, jnp.full((B, 1), float(t)))
    r0e = jax.vmap(lambda r, m: lm.as_repr(lm.as_lie(r) @ Lie.exp(sa * lm.as_tan(m)).inverse(), tb.a.repr_type))(rt, mu)
    print(f"  t={t:3d} sa={float(sa):.3f}  rt err {err(rt)}  r0_hat err {err(r0e)}")

  T = ns.timesteps
  cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))
  rt = jax.vmap(lambda r, zz: lm.as_repr(r @ Lie.exp(ns.sqrt_alphas[cur[0]] * zz), tb.a.repr_type))(r0_true, z)
  print("model sampler from GT-noised start:")
  for i, (t, tp) in enumerate(seq):
    key, k = jax.random.split(key)
    mu = head(f, rt, jnp.full((B, 1), float(t)))
    if os.environ.get("CANON") == "1":
      mu = jax.vmap(lambda m: lm.as_lie(m).log())(mu)  # principal-branch tangent
    r0e, rt = step(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
    if i % 20 == 0 or i == len(seq) - 1:
      print(f"  t={t:3d} r0 err {err(r0e)} rt err {err(rt)}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "model":
  import jaxlie
  model_check(sys.argv[1])


def path_check(exp, reps=4):
  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(exp, "ckpt"), int(os.environ.get("STEP", 800000))
  tb = R.get_testbed(a.lie_type)(a, exp, setup=True)
  lm, Lie, ns, st = tb.lie_metrics, tb.Lie, tb.noise_schedule, tb.state
  head = jax.jit(tb.create_head_apply())
  feat = jax.jit(tb.create_backbone_apply())
  data = next(iter(tb.create_train_dataset()))
  img = jnp.asarray(data["image"], jnp.float32) / 127.5 - 1.0
  rot = jnp.repeat(jnp.asarray(data["rotation"]), reps, 0)
  tran = jnp.repeat(jnp.asarray(data["translation"]), reps, 0)
  r0_true = jax.vmap(lambda q, t: Lie.from_rotation_and_translation(jaxlie.SO3(q), t))(rot, tran)
  B = rot.shape[0]
  z = jax.vmap(lambda k: tb.LieDist._sample_unit(k, n=()))(jax.random.split(jax.random.PRNGKey(0), B))
  zr = jax.vmap(lambda zz: lm.as_repr(zz, tb.a.repr_type))(z)
  f_eval = feat(img)
  for t in [99, 60, 20]:
    sa = ns.sqrt_alphas[t]
    rt = jax.vmap(lambda r, zz: lm.as_repr(r @ Lie.exp(sa * zz), tb.a.repr_type))(r0_true, z)
    tt = jnp.full((B, 1), t, jnp.int32)
    mu_train, _ = st.apply_fn({"params": st.params, "batch_stats": st.batch_stats}, img, rt, tt, mutable=["batch_stats"])
    mu_eval = head(f_eval, rt, tt)
    e = lambda m: float(jnp.median(jnp.linalg.norm(m - zr, axis=-1)))
    ch = jax.vmap(lm.get_distance_fn("chordal"))
    print(f"   |mu_v| {float(jnp.median(jnp.linalg.norm(mu_eval[:, :3], axis=-1))):.2f} |mu_w| {float(jnp.median(jnp.linalg.norm(mu_eval[:, 3:], axis=-1))):.2f}"
          f"  |z_v| {float(jnp.median(jnp.linalg.norm(zr[:, :3], axis=-1))):.2f} |z_w| {float(jnp.median(jnp.linalg.norm(zr[:, 3:], axis=-1))):.2f}"
          f"  chordal(mu,z) {float(jnp.median(ch(zr, mu_eval))):.3f}  chordal(0,z) {float(jnp.median(ch(zr, jnp.zeros_like(zr)))):.3f}")
    print(f"t={t:3d}  |mu-z| train-mode BN {e(mu_train):.3f}   eval-mode BN {e(mu_eval):.3f}   |z| {float(jnp.median(jnp.linalg.norm(zr, axis=-1))):.3f}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "path":
  import jaxlie
  path_check(sys.argv[1])


def shape_check(exp, n_img=16, reps=8):
  """Per shape on TEST images: one-shot r0 quality + wrap rate per t; sampler from GT-noised vs prior start.
  Rotation error = min geodesic to the symmetry-equivalent set (deg); translation = L2."""
  from liepose.data.symsolt.dataset import create_dataset
  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(exp, "ckpt"), int(os.environ.get("STEP", 800000))
  tb = R.get_testbed(a.lie_type)(a, exp, setup=True)
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=n_img, n_slices=reps)
  lm, Lie, ns = tb.lie_metrics, tb.Lie, tb.noise_schedule
  head, feat = jax.jit(tb.create_head_apply()), jax.jit(tb.create_backbone_apply())
  step = jax.jit(jax.vmap(tb.create_p_sample_fn()))
  sample_prior = jax.jit(jax.vmap(tb.create_sample_test_fn()))
  canon = jax.jit(jax.vmap(lambda m: lm.as_lie(m).log()))
  T = ns.timesteps
  cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))

  for shape in ["tet", "cube", "icosa", "cone", "cyl"]:
    ds = create_dataset("test-5k", [shape], include_equivalents=True, pad_equivalents=True)
    d = ds.get_batch(np.arange(n_img))
    img = jnp.asarray(d["image"], jnp.float32) / 127.5 - 1.0
    f = feat(img)
    rot = jnp.repeat(jnp.asarray(d["rotation"]), reps, 0)
    tran = jnp.repeat(jnp.asarray(d["translation"]), reps, 0)
    eq = np.repeat(np.asarray(jax.vmap(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix()))(jnp.asarray(d["rotations_equivalent"]))), reps, 0)
    r0_true = jax.vmap(lambda q, t: Lie.from_rotation_and_translation(jaxlie.SO3(q), t))(rot, tran)
    B = rot.shape[0]

    def err(r):
      g = jax.vmap(lm.as_lie)(r)
      Rm = np.asarray(jax.vmap(lambda x: x.rotation().as_matrix())(g))
      tr = np.einsum("bij,bkij->bk", Rm, eq)
      ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min(1)
      tt = np.asarray(jax.vmap(lambda x: x.translation())(g))
      return f"R {np.median(ang):6.2f}deg t {np.median(np.linalg.norm(tt - np.asarray(tran), axis=-1)):.3f}"

    z = jax.vmap(lambda k: tb.LieDist._sample_unit(k, n=()))(jax.random.split(jax.random.PRNGKey(1), B))
    ch = jax.vmap(lm.get_distance_fn("chordal"))
    print(f"== {shape}")
    for t in [99, 60, 20, 5]:
      sa = ns.sqrt_alphas[t]
      rt = jax.vmap(lambda r, zz: lm.as_repr(r @ Lie.exp(sa * zz), tb.a.repr_type))(r0_true, z)
      mu = head(f, rt, jnp.full((B, 1), float(t)))
      wrap = float((jnp.linalg.norm(mu[:, 3:], axis=-1) > np.pi).mean())
      mu_c = canon(mu)
      r0e = jax.vmap(lambda r, m: lm.as_repr(lm.as_lie(r) @ Lie.exp(sa * m).inverse(), tb.a.repr_type))(rt, mu_c)
      print(f"  one-shot t={t:2d}: chordal(mu,z) {float(jnp.median(ch(z, mu))):.3f} wrap {wrap:.2f} | rt {err(rt)} | r0_hat {err(r0e)}")
    for start in ["gt", "prior"]:
      key = jax.random.PRNGKey(2)
      if start == "gt":
        rt = jax.vmap(lambda r, zz: lm.as_repr(r @ Lie.exp(ns.sqrt_alphas[cur[0]] * zz), tb.a.repr_type))(r0_true, z)
      else:
        rt = sample_prior(jax.random.split(key, B))["rt"]
      for t, tp in seq:
        key, k = jax.random.split(key)
        mu = head(f, rt, jnp.full((B, 1), float(t)))
        r0e, rt = step(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      print(f"  sampler start={start:5s}: final {err(rt)}")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "shapes":
  import jaxlie
  shape_check(sys.argv[1])
