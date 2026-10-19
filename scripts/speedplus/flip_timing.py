"""Diagnostic (uses GT for measurement only): when do SPEED+ flips lock in during sampling, and how well does the
world-model energy separate good from flipped hypotheses at each noise level?

Runs sp_dit's sampler (uniform prior, 100-step DDIM) on a random subset of a split, n samples per image, and at
every step records the allocentric rotation error of the current hypothesis x_t against the GT allocentric rotation,
and the world-model energy of x_t (reference retrieved at x_t, as in the sampler). Reports, per step:
  share of samples with error > 90 deg; AUC of -energy for (error < 30) vs (error > 90);
and for samples that end flipped (> 90 deg at the end), the step after which they stay > 90 deg.

  DATADIR=... PYTHONPATH=src python scripts/speedplus/flip_timing.py --exp ... --step 100000 --out ...
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from liepose.exp.cmp.dit_ret import RetDenoiser
from liepose.exp.cmp.sp_dit import Testbed
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import crop_meta
from liepose.utils.exploration import Options


def auc(pos, neg):
  """P(score_pos > score_neg), ties half"""
  if len(pos) == 0 or len(neg) == 0:
    return None
  s = np.concatenate([pos, neg])
  r = s.argsort().argsort() + 1.0
  return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--exp", required=True)
  p.add_argument("--step", type=int, required=True)
  p.add_argument("--split", default="lightbox")
  p.add_argument("--n_img", type=int, default=500)
  p.add_argument("--n", type=int, default=4)
  p.add_argument("--out", required=True)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  cfg = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(a.exp, "configs.yaml")), resolve=True))
  cfg.restore_path, cfg.restore_step, cfg.skip_train = None, a.step, True
  tb = Testbed(cfg, a.exp, setup=True)
  assert int(tb.state.step) == a.step
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=100, n_slices=1)
  D = os.environ["DATADIR"]
  obs = TokenSplit(os.path.join(D, "speedplus_tokens"), a.split)
  meta, idx, _ = crop_meta(os.path.join(D, "speedplus_crops", a.split))
  assert (idx == obs.index).all()
  rows = np.sort(np.random.default_rng(0).choice(len(obs.obj), a.n_img, replace=False))

  model, poses = tb.model, tb._poses_fn()
  params = {"params": tb.state.params_ema if tb.a.ema_enabled else tb.state.params}
  p_fn = jax.vmap(tb.create_p_sample_fn())
  T = tb.noise_schedule.timesteps
  cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))
  lm, Lie = tb.lie_metrics, tb.Lie
  prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), tb.a.repr_type)))

  @jax.jit
  def step(key, rt, t, tp, Rg, tg, zr, Rr, tr, kr, z_obs, w, kg):
    B = rt.shape[0]
    key, k = jax.random.split(key)
    mu = model.apply(params, rt, Rg, tg, jnp.full((B, 1), t, jnp.float32), zr.astype(jnp.float32), Rr, tr, kr, z_obs, w, kg,
                     method=RetDenoiser.noise)
    _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
    return key, rt

  @jax.jit
  def energy(zr, Rr, tr, kr, Rg, tg, kg, z_obs, w):
    z_hat = model.apply(params, zr.astype(jnp.float32), Rr, tr, kr, Rg, tg, kg, method=RetDenoiser.render)
    return weighted_energy(z_hat, z_obs, w)

  ERR, EN, TS = [], [], [t for t, _ in seq]
  key = jax.random.PRNGKey(7)
  for s in range(0, len(rows), 25):
    r = np.repeat(rows[s:s + 25], a.n)
    tok, w, k = jnp.asarray(obs.tokens[r], jnp.float32), jnp.asarray(FG_FLOOR + obs.fg[r]), jnp.asarray(meta[r])
    Rgt = obs.R[r]  # GT allocentric rotation: measurement only
    key, k1, k2 = jax.random.split(key, 3)
    rt = prior(jax.random.split(k1, len(r)))
    e_seq, en_seq = [], []
    for t, tp in seq:
      Rg, tg = poses(rt)
      zr, Rr, tr, kr = tb._refs(np.asarray(Rg), np.asarray(tg), np.full(len(r), -1))
      en_seq.append(np.asarray(energy(zr, Rr, tr, kr, Rg, tg, k, tok, w)))
      c = np.clip((np.einsum("nij,nij->n", np.asarray(Rg, np.float64), Rgt) - 1) / 2, -1, 1)
      e_seq.append(np.degrees(np.arccos(c)))
      k2, rt = step(k2, rt, jnp.int32(t), jnp.int32(tp), Rg, tg, zr, Rr, tr, kr, tok, w, k)
    Rg, tg = poses(rt)
    c = np.clip((np.einsum("nij,nij->n", np.asarray(Rg, np.float64), Rgt) - 1) / 2, -1, 1)
    e_seq.append(np.degrees(np.arccos(c)))
    zr, Rr, tr, kr = tb._refs(np.asarray(Rg), np.asarray(tg), np.full(len(r), -1))
    en_seq.append(np.asarray(energy(zr, Rr, tr, kr, Rg, tg, k, tok, w)))
    ERR.append(np.stack(e_seq, 1))
    EN.append(np.stack(en_seq, 1))
    print(f"  {s + 25}/{len(rows)}", flush=True)
  ERR, EN = np.concatenate(ERR), np.concatenate(EN)  # (samples, 101): before each step, then final
  np.savez(os.path.join(a.out, f"{a.split}_flip_timing.npz"), err=ERR, energy=EN, t=np.array(TS + [-1]), rows=rows)

  final = ERR[:, -1]
  flip = final > 90
  # lock-in: first index after which the error stays > 90 deg
  above = ERR > 90
  stay = np.flip(np.cumprod(np.flip(above, 1), 1), 1).astype(bool)
  lock = np.where(flip, stay.argmax(1), -1)
  per_step = []
  for j in range(ERR.shape[1]):
    good, bad = ERR[:, j] < 30, ERR[:, j] > 90
    per_step.append({"i": j, "t": int(([*TS, -1])[j]), "share_gt90": float(bad.mean()), "share_lt30": float(good.mean()),
                     "auc_energy_good_vs_flip": auc(-EN[good, j], -EN[bad, j])})
  res = {"split": a.split, "n_img": int(len(rows)), "n_per_img": a.n, "final_flip_share": float(flip.mean()),
         "lock_in_index_quantiles_of_flips": [float(q) for q in np.percentile(lock[flip], [10, 25, 50, 75, 90])] if flip.any() else None,
         "per_step": per_step}
  json.dump(res, open(os.path.join(a.out, f"{a.split}_flip_timing.json"), "w"), indent=1)
  for d in per_step[::10] + [per_step[-1]]:
    print(json.dumps(d))
  print("final flip share", res["final_flip_share"], "lock-in step quantiles (10/25/50/75/90)", res["lock_in_index_quantiles_of_flips"])


if __name__ == "__main__":
  main()
