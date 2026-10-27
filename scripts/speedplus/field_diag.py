"""Where the feature field's error sits (diagnostic, GT pose and GT keypoint hull used; SPEED+ synthetic val).

Per crop: the field's rendering at the GT pose vs the crop's tokens, error per token split by the object's keypoint
hull at the GT pose (tok_alpha_oracle.hull_fg, 8x8, dilated); then for --per 1-3 deg perturbations of the GT
rotation, P(E_pert > E_gt) with three token weightings of the same squared error: uniform (the field's eval /
refinement), the GT hull (diagnostic), and the field's own rendered opacity at each candidate (GT-free).

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit:scripts/speedplus python scripts/speedplus/field_diag.py \
      --field $LOGS/probe/sp_field_v1 --out <json>
"""

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok_alpha_oracle import hull_fg  # noqa: E402

from liepose.exp.jepa_probe.field import FeatureField, FieldModel  # noqa: E402
from liepose.exp.jepa_probe.probe_frozen import perturb  # noqa: E402
from liepose.exp.jepa_probe.probe_tless import TokenSplit  # noqa: E402
from liepose.exp.jepa_probe.sp_geometry import SpeedPlusGeometry, crop_meta  # noqa: E402
from liepose.exp.jepa_probe.wm_eval import app_distance, app_stats  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--field", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--n", type=int, default=1500)
  p.add_argument("--per", type=int, default=8)
  p.add_argument("--batch", type=int, default=16)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  fa = SimpleNamespace(**json.load(open(os.path.join(a.field, "args.json"))))
  model = FieldModel(field=FeatureField(plane_res=fa.plane_res, plane_dim=fa.plane_dim, hid=getattr(fa, "hid", 128), radius=fa.radius,
                                        n_samp=fa.n_samp, geo=SpeedPlusGeometry()))
  obs = TokenSplit(os.path.join(D, "speedplus_tokens"), "val")
  meta, idx, _ = crop_meta(os.path.join(D, "speedplus_crops", "val"))
  assert (idx == obs.index).all()
  p0 = model.init(jax.random.PRNGKey(0), None, None, None, None, jnp.eye(3)[None], jnp.zeros((1, 3)), jnp.asarray(meta[:1]), jnp.zeros((1, 768)))
  params = jax.tree_util.tree_map(jnp.asarray, flax.serialization.from_bytes(p0, open(os.path.join(a.field, "params.msgpack"), "rb").read()))
  d_app = app_distance(D)
  rng = np.random.default_rng(0)
  rows = np.sort(rng.choice(len(obs.R), a.n, replace=False))
  sh = [np.load(f) for f in sorted(glob.glob(os.path.join(D, "speedplus_crops", "val", "shard_*.npz")))]
  dd = {k: np.concatenate([x[k] for x in sh]) for k in ["rotation", "translation", "dzi_size", "bbox_center"]}
  on = np.zeros(len(obs.R), bool)
  on[rows] = True
  hull = hull_fg(dd, on)  # (n, 64), rows in increasing order

  @jax.jit
  def render(R, t, k, z):
    return model.apply(params, None, None, None, None, R, t, k, app_stats(z), with_opacity=True)

  err_in, err_out, cov = [], [], []
  wins = {"uniform": [], "gt_hull (diagnostic)": [], "rendered_opacity": []}
  for s in range(0, a.n, a.batch):
    r = rows[s:s + a.batch]
    h = hull[s:s + a.batch]
    n, C = len(r), 1 + a.per
    Rc = np.concatenate([obs.R[r][:, None], perturb(np.repeat(obs.R[r], a.per, 0), 3.0, 1.0, rng).reshape(n, a.per, 3, 3)], 1)
    rep = lambda x: np.repeat(x, C, 0)
    z = jnp.asarray(obs.tokens[r], jnp.float32)
    pred, opa = render(jnp.asarray(Rc.reshape(-1, 3, 3), jnp.float32), jnp.asarray(rep(obs.t[r]), jnp.float32), jnp.asarray(rep(meta[r])),
                       jnp.repeat(z, C, 0))
    e = ((np.asarray(pred).reshape(n, C, 64, -1) - np.asarray(z)[:, None]) ** 2).mean(-1)  # (n, C, 64)
    opa = np.asarray(opa).reshape(n, C, 64)
    e0 = e[:, 0]
    err_in += list((e0 * h).sum(1) / np.maximum(h.sum(1), 1))
    err_out += list((e0 * (1 - h)).sum(1) / np.maximum((1 - h).sum(1), 1))
    cov += list(h.mean(1))
    for name, w in [("uniform", np.ones_like(e)), ("gt_hull (diagnostic)", np.repeat(h[:, None], C, 1) + 1e-3), ("rendered_opacity", opa + 1e-3)]:
      E = (w * e).sum(-1) / w.sum(-1)
      wins[name].append((E[:, 1:] > E[:, :1]).ravel())
  res = {"field": a.field, "n": a.n, "d_app": d_app, "hull_coverage_mean": float(np.mean(cov)),
         "gt_err_over_app_in_hull": float(np.median(np.sqrt(err_in)) / d_app),
         "gt_err_over_app_outside_hull": float(np.median(np.sqrt(err_out)) / d_app),
         "P 1-3": {k: float(np.concatenate(v).mean()) for k, v in wins.items()}}
  print(json.dumps(res, indent=1))
  json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
