"""Joint hypothesis scorer for T-LESS (FoundationPose-style selection, latent and symmetry-aware).

For each crop, the K LiePose samples become K tokens:
  world-model evidence  : log per-patch error (64) + log energy (1) + energy minus the set minimum (1)
  consensus (rotation)  : symmetry-aware angle to the other samples: mean, median, fraction within 5/15 deg
  consensus (transl.)   : distance to the set-median translation, mean pairwise translation distance
A 2-layer transformer attends across the K tokens and outputs one score each. Loss: cross-entropy against
soft targets softmax(-(rot_err/5 + tran_err/0.1)). Trained on LiePose samples drawn on training crops; the
checkpoint is chosen on a held-out 10% of those training crops (the test set is only evaluated).

  PYTHONPATH=src python scripts/tless/scorer.py --train_samples train_samples.npz --test_samples test_samples.npz \
      --wm $LOGS/probe/tless_v1 --tokens $DATADIR/tless_tokens --out $LOGS/tless_scorer
"""

import argparse
import json
import os

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from liepose.exp.jepa_probe.probe import pose_feat
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.probe_tless import TemplatePoseModel, load_symmetries

PICKS = ["random", "medoid", "min_energy", "scorer", "oracle"]


def load_wm(wm_dir, tok_shape):
  wa = json.load(open(os.path.join(wm_dir, "args.json")))
  objs = np.load(os.path.join(wm_dir, "objects.npy"))
  wm = TemplatePoseModel(n_obj=len(objs), n_tok=tok_shape[0], tok_dim=tok_shape[1], dim=wa["dim"], depth=wa["depth"])
  init = wm.init(jax.random.PRNGKey(0), jnp.zeros((1,), jnp.int32), jnp.zeros((1, 9)))
  params = flax.serialization.from_bytes(init, open(os.path.join(wm_dir, "params.msgpack"), "rb").read())
  return wm, jax.tree_util.tree_map(jnp.asarray, params), {int(o): i for i, o in enumerate(objs)}


def featurize(S, tok, fg, rows, wm, wm_params, obj2idx, syms, chunk=64):
  """S: samples dict (R (N,K,3,3), t (N,K,3), obj (N,)); rows: token row per crop. Returns (N,K,F) features."""
  N, K = S["R"].shape[:2]

  @jax.jit
  def wm_feats(obj, g, tk, w):
    pred = wm.apply(wm_params, obj, g)
    per = ((pred - tk.astype(jnp.float32)) ** 2).mean(-1)  # (B,64)
    ww = FG_FLOOR + w
    e = (ww * per).sum(-1) / ww.sum(-1)
    return jnp.log(per + 1e-4), jnp.log(e + 1e-6)

  F = []
  for i0 in range(0, N, chunk):
    ii = np.arange(i0, min(N, i0 + chunk))
    B = len(ii)
    g = pose_feat(S["R"][ii].reshape(-1, 3, 3), S["t"][ii].reshape(-1, 3))
    obj = np.repeat([obj2idx[int(o)] for o in S["obj"][ii]], K)
    tk = np.repeat(np.asarray(tok[rows[ii]]), K, axis=0)
    w = np.repeat(np.asarray(fg[rows[ii]], np.float32), K, axis=0)
    lp, le = map(np.asarray, wm_feats(jnp.asarray(obj), jnp.asarray(g), jnp.asarray(tk), jnp.asarray(w)))
    lp, le = lp.reshape(B, K, -1), le.reshape(B, K)
    for b, i in enumerate(ii):
      R, t = S["R"][i], S["t"][i]
      Sy = syms[int(S["obj"][i])]
      # symmetry-aware pairwise rotation angle: min over S of angle(R_a, R_b S)
      tr = np.einsum("aij,bmij->abm", R, np.einsum("bij,mjk->bmik", R, Sy))
      ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min(-1)
      np.fill_diagonal(ang, np.nan)
      dt = np.linalg.norm(t[:, None] - t[None], axis=-1)
      np.fill_diagonal(dt, np.nan)
      f = np.concatenate([
        lp[b], le[b][:, None], (le[b] - le[b].min())[:, None],
        np.nanmean(ang, 1)[:, None] / 30, np.nanmedian(ang, 1)[:, None] / 30,
        np.nanmean(ang < 5, 1)[:, None], np.nanmean(ang < 15, 1)[:, None],
        np.linalg.norm(t - np.median(t, 0), axis=-1)[:, None], np.nanmean(dt, 1)[:, None],
      ], axis=-1)
      F.append(f)
  return np.stack(F).astype(np.float32)


class Scorer(nn.Module):
  dim: int = 128
  depth: int = 2
  heads: int = 4

  @nn.compact
  def __call__(self, x):  # (B,K,F)
    h = nn.Dense(self.dim)(nn.LayerNorm()(x))
    for _ in range(self.depth):
      y = nn.LayerNorm()(h)
      h = h + nn.MultiHeadDotProductAttention(num_heads=self.heads)(y, y)
      y = nn.LayerNorm()(h)
      h = h + nn.Dense(self.dim)(nn.gelu(nn.Dense(self.dim * 2)(y)))
    return nn.Dense(1)(nn.LayerNorm()(h))[..., 0]


def soft_target(rot_err, tran_err):
  return jax.nn.softmax(-(rot_err / 5.0 + tran_err / 0.1), axis=-1)


def picks_table(S, scores, F):
  re, te = S["rot_err"], S["tran_err"]
  N = len(re)
  rows = np.arange(N)
  # medoid: most symmetry-aware agreement (feature col = mean pairwise angle)
  med = F[:, :, 66].argmin(1)
  idx = {"random": np.zeros(N, int), "medoid": med, "min_energy": F[:, :, 64].argmin(1),
         "scorer": scores.argmax(1), "oracle": np.argmin(re / 5.0 + te / 0.1, 1)}
  out = {}
  for k, j in idx.items():
    r, t = re[rows, j], te[rows, j]
    out[k] = {"rot_mean": float(r.mean()), "rot_median": float(np.median(r)), "rot_lt5": float(100 * (r < 5).mean()),
              "rot_lt10": float(100 * (r < 10).mean()), "rot_gt30": float(100 * (r > 30).mean()),
              "tran_mean": float(t.mean()), "tran_median": float(np.median(t))}
  return out


def print_table(title, tab):
  print(f"\n== {title}")
  print(f"{'pick':12s} {'R mean':>7s} {'R med':>6s} {'R<5':>6s} {'R<10':>6s} {'>30deg':>7s} {'t mean':>7s} {'t med':>6s}")
  for k in PICKS:
    v = tab[k]
    print(f"{k:12s} {v['rot_mean']:7.2f} {v['rot_median']:6.2f} {v['rot_lt5']:6.1f} {v['rot_lt10']:6.1f} {v['rot_gt30']:7.2f} "
          f"{v['tran_mean']:7.4f} {v['tran_median']:6.4f}", flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--train_samples", required=True)
  p.add_argument("--test_samples", required=True)
  p.add_argument("--wm", required=True)
  p.add_argument("--tokens", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--steps", type=int, default=6000)
  p.add_argument("--batch", type=int, default=128)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--seed", type=int, default=0)
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  syms = load_symmetries(os.path.join(os.environ["DATADIR"], "bop_datasets/tless/models_eval/models_info.json"))

  tr_tok = np.load(os.path.join(a.tokens, "train_tokens.npy"), mmap_mode="r")
  tr_fg = np.load(os.path.join(a.tokens, "train_fg.npy"), mmap_mode="r")
  te_tok = np.load(os.path.join(a.tokens, "test_tokens.npy"), mmap_mode="r")
  te_fg = np.load(os.path.join(a.tokens, "test_fg.npy"), mmap_mode="r")
  te_row_of = {int(ix): r for r, ix in enumerate(np.load(os.path.join(a.tokens, "test_meta.npz"))["index"])}
  wm, wm_params, obj2idx = load_wm(a.wm, tr_tok.shape[1:])

  Str = dict(np.load(a.train_samples))
  Ste = dict(np.load(a.test_samples))
  feat_path = os.path.join(a.out, "features.npz")
  if os.path.isfile(feat_path):
    ff = np.load(feat_path)
    Ftr, Fte = ff["train"], ff["test"]
  else:
    Ftr = featurize(Str, tr_tok, tr_fg, Str["row"], wm, wm_params, obj2idx, syms)
    Fte = featurize(Ste, te_tok, te_fg, np.array([te_row_of[int(i)] for i in Ste["index"]]), wm, wm_params, obj2idx, syms)
    np.savez(feat_path, train=Ftr, test=Fte)
  print("features", Ftr.shape, Fte.shape, flush=True)

  rng = np.random.default_rng(a.seed)
  perm = rng.permutation(len(Ftr))
  n_val = len(Ftr) // 10
  val, trn = perm[:n_val], perm[n_val:]
  model = Scorer()
  params = model.init(jax.random.PRNGKey(a.seed), jnp.zeros((1,) + Ftr.shape[1:]))
  tx = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, a.lr, 300, a.steps, a.lr * 0.05), weight_decay=0.05)
  opt_state = tx.init(params)

  @jax.jit
  def step(params, opt_state, x, re, te):
    def loss_fn(p):
      s = model.apply(p, x)
      return optax.softmax_cross_entropy(s, soft_target(re, te)).mean()
    loss, g = jax.value_and_grad(loss_fn)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss

  score_fn = jax.jit(lambda p, x: model.apply(p, x))

  def scores(p, F):
    return np.concatenate([np.asarray(score_fn(p, jnp.asarray(F[i : i + 1024]))) for i in range(0, len(F), 1024)])

  Sval = {k: v[val] for k, v in Str.items() if k in ("rot_err", "tran_err")}
  best, best_params = None, params
  for it in range(1, a.steps + 1):
    b = rng.choice(trn, a.batch)
    kp = rng.permuted(np.tile(np.arange(Ftr.shape[1]), (a.batch, 1)), axis=1)  # random sample order
    x = np.take_along_axis(Ftr[b], kp[..., None], 1)
    re = np.take_along_axis(Str["rot_err"][b], kp, 1)
    te = np.take_along_axis(Str["tran_err"][b], kp, 1)
    params, opt_state, loss = step(params, opt_state, jnp.asarray(x), jnp.asarray(re), jnp.asarray(te))
    if it % 500 == 0:
      tab = picks_table(Sval, scores(params, Ftr[val]), Ftr[val])["scorer"]
      crit = tab["rot_mean"] / 5 + tab["tran_mean"] / 0.1
      print(f"it {it} loss {float(loss):.4f}  val scorer R mean {tab['rot_mean']:.2f} >30 {tab['rot_gt30']:.2f}% t {tab['tran_mean']:.4f}", flush=True)
      if best is None or crit < best:
        best, best_params = crit, params
  with open(os.path.join(a.out, "scorer.msgpack"), "wb") as f:
    f.write(flax.serialization.to_bytes(best_params))
  tab_val = picks_table(Sval, scores(best_params, Ftr[val]), Ftr[val])
  tab_te = picks_table(Ste, scores(best_params, Fte), Fte)
  print_table("held-out TRAIN crops (model selection)", tab_val)
  print_table(f"TEST crops (n={len(Fte)}, K={Fte.shape[1]})", tab_te)
  json.dump({"val": tab_val, "test": tab_te}, open(os.path.join(a.out, "results.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
