"""Training-free world-model guidance for a frozen LiePose checkpoint on SYMSOL-T.

After each reverse step with t >= --gate_t, every sample takes a step down the latent energy
E(g) = weighted ||P(z_ref, g) - z_obs|| of the frozen probe-v3 world model:
    r <- r · exp(-lam * sap * ∇E / |∇E|)       (gradient w.r.t. a right-perturbation in se(3))
lam = 0 is the unguided control (same seeds). Optionally K samples per image + min-energy pick.

  DATADIR=... PYTHONPATH=src python scripts/guidance_eval.py <liepose_exp> --wm <probe_dir> --cache <dino_cache>
"""

import argparse
import json
import os

import flax
import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from liepose.exp.cmp.testbed import TokenCache, pose_feat_jnp
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, FrozenPoseModel, pool_tokens
from liepose.exp.symsolt import run as R
from liepose.utils.exploration import Options

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("exp")
  p.add_argument("--wm", required=True)
  p.add_argument("--cache", required=True)
  p.add_argument("--step", type=int, default=800000)
  p.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
  p.add_argument("--gate_t", type=int, default=40)
  p.add_argument("--per_shape", type=int, default=500)
  p.add_argument("--k", type=int, default=1, help="samples per image; >1 adds min-energy selection")
  p.add_argument("--out", required=True)
  c = p.parse_args()

  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(c.exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(c.exp, "ckpt"), c.step
  tb = R.get_testbed(a.lie_type)(a, c.exp, setup=True)
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=100, n_slices=c.k)
  lm, Lie, ns, st = tb.lie_metrics, tb.Lie, tb.noise_schedule, tb.state

  # frozen world model
  with open(os.path.join(c.wm, "args.json")) as f:
    pa = json.load(f)
  refs, _ = pool_tokens(np.load(os.path.join(c.cache, "refs_tokens.npy")), np.load(os.path.join(c.cache, "refs_fg.npy")), 2)
  refs = jnp.asarray(refs, jnp.float32)
  wm = FrozenPoseModel(dim=pa["dim"], depth=pa["depth"], heads=pa["dim"] // 64, out_dim=refs.shape[-1])
  init = wm.init(jax.random.PRNGKey(0), jnp.zeros((1,) + refs.shape[1:]), jnp.zeros((1, 9)))
  with open(os.path.join(c.wm, "params.msgpack"), "rb") as f:
    wm_params = flax.serialization.from_bytes(init, f.read())

  def energy_one(g, shape, tok, fg):
    pred = wm.apply(wm_params, refs[shape][None], pose_feat_jnp(g.rotation().as_matrix()[None], g.translation()[None]))[0]
    w = FG_FLOOR + fg
    return (w * ((pred - tok) ** 2).mean(-1)).sum() / w.sum()

  def energy_and_grad(r, shape, tok, fg):
    g = lm.as_lie(r)
    f = lambda xi: energy_one(g @ Lie.exp(xi), shape, tok, fg)
    return jax.value_and_grad(f)(jnp.zeros(6))

  eg = jax.vmap(energy_and_grad)
  backbone = jax.jit(tb.create_backbone_apply())
  head = tb.create_head_apply()
  p_fn = jax.vmap(tb.create_p_sample_fn())
  prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), a.repr_type)))
  T = ns.timesteps
  cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
  seq = jnp.asarray(list(zip(cur, cur[1:] + [0])), jnp.int32)

  @jax.jit
  def sample(key, feat, rt, shape, tok, fg, lam):
    B = rt.shape[0]

    def step(carry, ttp):
      rt, key = carry
      t, tp = ttp[0], ttp[1]
      key, k = jax.random.split(key)
      mu = head(feat, rt, jnp.full((B, 1), t, jnp.float32))
      _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      _, gr = eg(rt, shape, tok, fg)
      d = gr / (jnp.linalg.norm(gr, axis=-1, keepdims=True) + 1e-8)
      scale = jnp.where(t >= c.gate_t, lam * ns.sqrt_alphas_prev[tp], 0.0)
      rt = jax.vmap(lambda r, dd: lm.as_repr(lm.as_lie(r) @ Lie.exp(-scale * dd), a.repr_type))(rt, d)
      return (rt, key), None

    (rt, _), _ = jax.lax.scan(step, (rt, key), seq)
    e, _ = eg(rt, shape, tok, fg)
    return rt, e

  @jax.jit
  def errors(rt, eq, t_gt):
    g = jax.vmap(lm.as_lie)(rt)
    Rm = jax.vmap(lambda x: x.rotation().as_matrix())(g)
    Req = jax.vmap(jax.vmap(lambda q: jaxlie.SO3(q).as_matrix()))(eq)
    ang = jnp.degrees(jnp.arccos(jnp.clip((jnp.einsum("bij,bkij->bk", Rm, Req) - 1) / 2, -1, 1))).min(1)
    return ang, jnp.linalg.norm(jax.vmap(lambda x: x.translation())(g) - t_gt, axis=-1)

  cache = TokenCache(c.cache, "test-5k")
  test = tb.create_test_dataset()
  lab = np.array([x["label_shape"] for x in test.dataset.annos])
  sel = np.concatenate([np.nonzero(lab == s)[0][: c.per_shape] for s in range(5)])
  batches = np.array_split(sel, np.arange(0, len(sel), 100))[1:]

  results = {}
  for lam in c.lams:
    key = jax.random.PRNGKey(7)  # identical noise for every lam
    rot, tran, en, labels = [], [], [], []
    for bi in tqdm(batches, desc=f"lam={lam}", leave=False):
      data = test.dataset.get_batch(bi)
      k = c.k
      img = jnp.asarray(data["image"], jnp.float32) / 127.5 - 1.0
      feat = backbone(img)
      tok, fg = cache.gather(data["index"], data["label_shape"])
      rep = lambda x: jnp.repeat(jnp.asarray(x), k, 0)
      key, k1, k2 = jax.random.split(key, 3)
      rt0 = prior(jax.random.split(k1, len(bi) * k))
      rt, e = sample(k2, feat, rt0, rep(data["label_shape"]), rep(tok).astype(jnp.float32), rep(fg), lam)
      ang, te = errors(rt, rep(data["rotations_equivalent"]), rep(data["translation"]))
      rot.append(np.asarray(ang).reshape(-1, k))
      tran.append(np.asarray(te).reshape(-1, k))
      en.append(np.asarray(e).reshape(-1, k))
      labels.append(np.asarray(data["label_shape"]))
    rot, tran, en, labels = map(np.concatenate, (rot, tran, en, labels))
    rows = np.arange(len(rot))
    picks = {"first": (rot[:, 0], tran[:, 0])}
    if c.k > 1:
      j = en.argmin(1)
      picks["min_energy"] = (rot[rows, j], tran[rows, j])
    results[str(lam)] = {}
    for pk, (r_, t_) in picks.items():
      for si, name in enumerate(SHAPES + ["all"]):
        m = labels == si if name != "all" else np.ones(len(labels), bool)
        results[str(lam)][f"{pk}/{name}"] = {"rot_mean": float(r_[m].mean()), "rot_median": float(np.median(r_[m])),
                                            "gt10_pct": float(100 * (r_[m] > 10).mean()), "tran_mean": float(t_[m].mean())}
    print(f"lam={lam}", {k_: {kk: round(vv, 3) for kk, vv in v.items()} for k_, v in results[str(lam)].items() if k_.endswith("/all")}, flush=True)
  with open(c.out, "w") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  main()
