"""Branch-and-select inside the diffusion loop, for an sp_field_dit checkpoint: at each of the first --branch_frac of the
sampling steps, every chain draws --K stochastic proposals of its next pose (the sampler's own noise), the frozen field
renders each proposal in the observation's crop, and the chain keeps the proposal of lowest energy (paper eq:energy,
the foreground-weighted squared token distance). The remaining steps are the plain sampler; K=1 is the plain sampler. --smc_lambda > 0: sequential Monte Carlo instead,
reweighting and resampling the N chains of a crop by exp(-lambda E) of the sampler's predicted clean pose (no extra denoiser call).
N chains per crop from independent priors, saved as <split>_samples.npz for consensus_score.py (sample 0 = one sample,
and the consensus pick). Inference reads only the crop (tokens / images) and its box meta; GT is read by the scorer only.
Settings are chosen on the synthetic val split; lightbox / sunlamp are run once with the chosen settings.

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit:scripts/speedplus python scripts/speedplus/test_branch.py \
      --den_mode field_full_lat_ft --field_dir $LOGS/probe/sp_field_v1 --restore $LOGS/ffl_ft_ckpt --step 100000 \
      --K 8 --branch_frac 0.3 --splits val --limit 300 --out <dir>
"""

import argparse
import glob
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esa_score import allo_to_ego  # noqa: E402

from liepose.data.speedplus.dataset import recover_translation  # noqa: E402
from liepose.exp.cmp import sp_field_dit  # noqa: E402
from liepose.exp.cmp.sp_field_dit import FtFieldDenoiser  # noqa: E402
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR  # noqa: E402
from liepose.exp.jepa_probe.probe_tless import TokenSplit  # noqa: E402
from liepose.exp.jepa_probe.sp_geometry import crop_meta  # noqa: E402
from liepose.utils.config import get_args, load_config_from_args  # noqa: E402
from liepose.utils.exploration import Options  # noqa: E402


def create_branch_sampler(tb, n_steps, K, n_branch, S=16, lam=0.0):
  """tb.create_sampler with K proposals per chain and energy selection at the first n_branch steps, or (lam > 0) SMC over
  the S chains of each crop with the potential exp(-lam E(predicted clean pose)) at the first n_branch steps"""
  model, poses = tb.model, tb._poses_fn()
  params = {"params": tb.state.params_ema if tb.a.ema_enabled else tb.state.params}
  p_fn = jax.vmap(tb.create_p_sample_fn())
  T = tb.noise_schedule.timesteps
  cur = (np.linspace(T, 0, n_steps, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))

  def energy(rt, z_obs, w, kg):
    Rg, tg = poses(rt)
    z_hat, _, _ = tb._render(Rg, tg, kg, z_obs)
    e = ((z_hat - z_obs) ** 2).mean(-1)
    return (w * e).sum(-1) / w.sum(-1)

  def denoise(rt, t, z_obs, w, kg, z_ext):
    B = rt.shape[0]
    Rg, tg = poses(rt)
    z_hat, opa, h = tb._render(Rg, tg, kg, z_obs)
    return model.apply(params, rt, jnp.full((B, 1), t, jnp.float32), kg, z_obs, w, z_hat, opa, h, None, z_ext)

  @jax.jit
  def step(key, rt, t, tp, z_obs, w, kg, z_ext):
    B = rt.shape[0]
    key, k = jax.random.split(key)
    mu = denoise(rt, t, z_obs, w, kg, z_ext)
    _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
    return key, rt

  @jax.jit
  def step_branch(key, rt, t, tp, z_obs, w, kg, z_ext):
    B = rt.shape[0]
    key, k = jax.random.split(key)
    mu = denoise(rt, t, z_obs, w, kg, z_ext)

    def propose(kk):  # one proposal per chain from the sampler's own noise, and its energy
      _, r = p_fn(jax.random.split(kk, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      return r, energy(r, z_obs, w, kg)

    props, E = jax.lax.map(propose, jax.random.split(k, K))  # (K, B, ...), (K, B)
    best = jnp.argmin(E, 0)
    return key, jax.tree_util.tree_map(lambda x: x[best, jnp.arange(B)], props)

  @jax.jit
  def step_smc(key, rt, t, tp, z_obs, w, kg, z_ext, logw, e_prev):
    """one sampler step, then SMC reweighting of the S chains of each crop by the potential exp(-lam E(r0)), r0 = the
    sampler's prediction of the clean pose at this step (incremental weight G_t / G_{t+1}); resampling (systematic,
    within each crop's chains) when the effective sample size falls below S / 2"""
    B = rt.shape[0]
    G = B // S
    key, k, kr = jax.random.split(key, 3)
    mu = denoise(rt, t, z_obs, w, kg, z_ext)
    r0, rp = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
    e = energy(r0, z_obs, w, kg)
    logw = logw - lam * (e - e_prev)
    lw = logw.reshape(G, S)
    lw = lw - jax.scipy.special.logsumexp(lw, 1, keepdims=True)
    ess = 1.0 / jnp.exp(2 * lw).sum(1)  # (G,)
    u = (jax.random.uniform(kr, (G, 1)) + jnp.arange(S)[None]) / S
    idx = jax.vmap(jnp.searchsorted)(jnp.cumsum(jnp.exp(lw), 1), u).clip(0, S - 1)  # (G, S)
    res = (ess < S / 2)[:, None]
    idx = jnp.where(res, idx, jnp.arange(S)[None])
    gi = (idx + jnp.arange(G)[:, None] * S).reshape(-1)
    rp = jax.tree_util.tree_map(lambda x: x[gi], rp)
    logw = jnp.where(res, 0.0, lw).reshape(-1)
    return key, rp, logw, e[gi], res.mean()

  encode = jax.jit(lambda image: model.apply(params, image, method=FtFieldDenoiser.encode)) if tb.ft else None
  stats = {"resample_rate": []}

  def sample(key, rt, tok, fg, meta, image=None):
    z_ext = None
    if image is not None:
      tok, z_ext = encode(image)
    if lam > 0:
      B = rt.shape[0]
      logw, e_prev = jnp.zeros(B), jnp.zeros(B)
      for i, (t, tp) in enumerate(seq):
        if i < n_branch:
          key, rt, logw, e_prev, rr = step_smc(key, rt, jnp.int32(t), jnp.int32(tp), tok, fg, meta, z_ext, logw, e_prev)
          stats["resample_rate"].append(float(rr))
        else:
          key, rt = step(key, rt, jnp.int32(t), jnp.int32(tp), tok, fg, meta, z_ext)
      return rt
    for i, (t, tp) in enumerate(seq):
      f = step_branch if (K > 1 and i < n_branch) else step
      key, rt = f(key, rt, jnp.int32(t), jnp.int32(tp), tok, fg, meta, z_ext)
    return rt

  sample.stats = stats
  return sample


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--den_mode", required=True)
  p.add_argument("--field_dir", required=True)
  p.add_argument("--restore", required=True)
  p.add_argument("--step", type=int, required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--splits", nargs="+", default=["lightbox", "sunlamp"])
  p.add_argument("--n_samples", type=int, default=16)
  p.add_argument("--K", type=int, default=8, help="proposals per chain at a branching step (1 = plain sampler)")
  p.add_argument("--branch_frac", type=float, default=0.3, help="share of the sampling steps, from the noisy end, that branch")
  p.add_argument("--steps", type=int, default=100)
  p.add_argument("--smc_lambda", type=float, default=0.0, help="> 0: SMC over each crop's chains instead of best-of-K")
  p.add_argument("--crops_per_batch", type=int, default=8)
  p.add_argument("--limit", type=int, default=0, help="first n crops (val sweeps; consensus_score.py scores what is saved)")
  p.add_argument("--no_ema", action="store_true", help="sample with the live weights (ema_check.py's choice)")
  a = p.parse_args()
  sys.argv = [sys.argv[0], "--root_path", os.path.join(a.out, "_tb"), f"den_mode=[{a.den_mode}]", f"field_dir=[{a.field_dir}]",
              "train_steps=[100000]", "batch_size=[64]", "n_slices=[64]", "skip_train=[True]", f"restore_path=[{a.restore}]",
              f"restore_step=[{a.step}]"] + (["ema_enabled=[False]"] if a.no_ema else [])
  ga = get_args(exp_name="branch", desc="branch-and-select")
  opt = next(iter(load_config_from_args(ga, sp_field_dit.build_explorations()).expand()))
  opt.pop("sample_options", None)
  tb = sp_field_dit.Testbed(opt, os.path.join(a.out, "_tb"), setup=True)
  assert int(tb.state.step) in (a.step, a.step + 1), int(tb.state.step)
  tb.opt = Options(dict(use_ddpm=False, eta=1.0, beta_skip=True, steps=a.steps, batch_size=100, n_slices=1))
  D = os.environ["DATADIR"]
  lm, Lie = tb.lie_metrics, tb.Lie
  prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), tb.a.repr_type)))
  n_branch = int(round(a.branch_frac * a.steps))
  S = a.n_samples
  sample, poses = create_branch_sampler(tb, a.steps, a.K, n_branch, S, a.smc_lambda), tb._poses_fn()
  os.makedirs(a.out, exist_ok=True)
  info = {"args": vars(a), "n_branch_steps": n_branch, "ema": bool(tb.a.ema_enabled)}
  for split in a.splits:
    obs = TokenSplit(os.path.join(D, "speedplus_tokens"), split)
    meta, idx, _ = crop_meta(os.path.join(D, "speedplus_crops", split))
    assert (idx == obs.index).all()
    sh = sorted(glob.glob(os.path.join(D, "speedplus_crops", split, "shard_*.npz")))
    ims = np.concatenate([np.load(f)["image"] for f in sh]) if tb.ft else None
    box = {k: np.concatenate([np.load(f)[k] for f in sh]) for k in ["bbox", "bbox_center", "focal_ratio"]}
    key = jax.random.PRNGKey(tb.a.seed + 5)
    Rall, tall = [], []
    n = len(obs.index) if not a.limit else min(a.limit, len(obs.index))
    t0 = time.time()
    for s in range(0, n, a.crops_per_batch):
      rows = np.repeat(np.arange(s, min(s + a.crops_per_batch, n)), S)
      key, k1, k2 = jax.random.split(key, 3)
      rt = sample(k2, prior(jax.random.split(k1, len(rows))), None if tb.ft else jnp.asarray(obs.tokens[rows], jnp.float32),
                  jnp.asarray(FG_FLOOR + obs.fg[rows]), jnp.asarray(meta[rows]), None if ims is None else jnp.asarray(ims[rows], jnp.float32))
      Rg, tg = poses(rt)
      Rall.append(np.asarray(Rg, np.float64))
      tall.append(np.asarray(tg, np.float64))
      if (s // a.crops_per_batch) % 20 == 0:
        print(f"  {split} {min(s + a.crops_per_batch, n)}/{n}  {time.time() - t0:.0f}s", flush=True)
    R_allo, t_sc = np.concatenate(Rall), np.concatenate(tall)
    rep = lambda x: np.repeat(x[:n], S, 0)
    t_est = recover_translation(t_sc, rep(box["bbox"]), rep(box["bbox_center"]), rep(box["focal_ratio"]), tb.a.tran_scale, tb.a.tran_offset)
    R_est = allo_to_ego(R_allo, t_est)
    np.savez(os.path.join(a.out, f"{split}_samples.npz"), index=obs.index[:n].astype(np.int64), R=R_est.reshape(n, S, 3, 3), t=t_est.reshape(n, S, 3))
    rr = sample.stats["resample_rate"]
    info[split] = {"n": int(n), "seconds": time.time() - t0, "mean_resample_rate": float(np.mean(rr)) if rr else 0.0}
    rr.clear()
    print(f"{split}: saved {n} x {S} samples in {time.time() - t0:.0f}s", flush=True)
  json.dump(info, open(os.path.join(a.out, "branch_info.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
