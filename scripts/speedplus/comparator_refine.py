"""Gate of the learned comparator (liepose.exp.jepa_probe.comparator) on the SPEED+ synthetic val split, full split.

  1 bias (diagnostic, starts AT the true pose): --iters comparator steps from the GT pose; rotation drift (median).
    Gate <= 0.3 deg (the field energy's minimum sits 1.4 deg away: field_refine.py).
  2 pipeline (GT-free): LiePose's 16-sample consensus (select_wm2.py's rule: the sample with the most others within
    10 deg) -> --iters comparator steps; official ESA (esa_score.esa). Gate: ESA and median rotation error both below
    consensus. Inference reads the crop's tokens, box meta and the frozen field; GT is read by the scorer only.
  3 convergence (added after a smoke model that outputs ~0 passed 1 and 2 by doing nothing, before any full-model
    result): from GT perturbed by 1-3 deg, median rotation error after refinement <= 0.5 deg.
Also reported: convergence from GT perturbed by 1-3 / 3-10 deg (diagnostic).

  DATADIR=... SPEEDPLUS_ROOT=... PYTHONPATH=src:lib/bop_toolkit:scripts/speedplus python scripts/speedplus/comparator_refine.py \
      --cmp $LOGS/probe/sp_cmp_v1 --samples <dir>/val_samples.npz --out <json>
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
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esa_score import allo_to_ego, esa  # noqa: E402

from liepose.data.speedplus.dataset import recover_translation  # noqa: E402
from liepose.exp.jepa_probe.comparator import Comparator, load_field, make_refine  # noqa: E402
from liepose.exp.jepa_probe.probe_frozen import perturb  # noqa: E402
from liepose.exp.jepa_probe.probe_tless import TokenSplit  # noqa: E402
from liepose.exp.jepa_probe.sp_geometry import crop_meta  # noqa: E402
from liepose.exp.speedplus.run import TRAN_OFFSET, TRAN_SCALE  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--cmp", required=True)
  p.add_argument("--samples", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--iters", type=int, default=4)
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--limit", type=int, default=0)
  a = p.parse_args()
  D = os.environ["DATADIR"]
  ca = SimpleNamespace(**json.load(open(os.path.join(a.cmp, "args.json"))))
  field_model, field_params = load_field(ca.field)
  cmp = Comparator(dim=ca.dim, depth=ca.depth)
  obs = TokenSplit(os.path.join(D, "speedplus_tokens"), "val")
  meta, idx, _ = crop_meta(os.path.join(D, "speedplus_crops", "val"))
  assert (idx == obs.index).all()
  c0 = cmp.init(jax.random.PRNGKey(0), jnp.zeros((1, 64, 384)), jnp.zeros((1, 64, 384)), jnp.zeros((1, 64)), jnp.asarray(meta[:1]))
  cp = jax.tree_util.tree_map(jnp.asarray, flax.serialization.from_bytes(c0, open(os.path.join(a.cmp, "params.msgpack"), "rb").read()))
  refine = make_refine(cmp, field_model, field_params, a.iters, ca.t_unit)
  n = len(obs.R) if not a.limit else a.limit

  sm = np.load(a.samples)
  pos = {int(i): r for r, i in enumerate(sm["index"])}
  srow = np.array([pos[int(i)] for i in obs.index[:n]])
  Re_s = sm["R"][srow]
  G = np.degrees(np.arccos(np.clip((np.einsum("nsij,ntij->nst", Re_s, Re_s) - 1) / 2, -1, 1)))
  cons = ((G < 10).sum(-1) - 1e-3 * G.mean(-1)).argmax(1)
  R0 = Rotation.from_quat(np.roll(sm["quat_allo"][srow, cons], -1, -1)).as_matrix()
  t0 = sm["t_scaled"][srow, cons].astype(np.float64)
  rng = np.random.default_rng(3)
  starts = {"from_gt (diagnostic)": (obs.R[:n], obs.t[:n]), "from_gt+1-3deg (diagnostic)": (perturb(obs.R[:n], 3.0, 1.0, rng), obs.t[:n]),
            "from_gt+3-10deg (diagnostic)": (perturb(obs.R[:n], 10.0, 3.0, rng), obs.t[:n]), "from_consensus": (R0, t0)}
  out = {}
  for name, (Rs, tsc) in starts.items():
    Rr, tr = [], []
    for s in range(0, n, a.batch):
      r = np.arange(s, min(s + a.batch, n))
      R1, t1 = refine(cp, jnp.asarray(Rs[r], jnp.float32), jnp.asarray(tsc[r], jnp.float32), jnp.asarray(meta[r]), jnp.asarray(obs.tokens[r], jnp.float32))
      Rr.append(np.asarray(R1, np.float64))
      tr.append(np.asarray(t1, np.float64))
    out[name] = (np.concatenate(Rr), np.concatenate(tr))
    print("refined", name, flush=True)

  # scorer only
  sh = [np.load(f) for f in sorted(glob.glob(os.path.join(D, "speedplus_crops", "val", "shard_*.npz")))]
  box = {k: np.concatenate([x[k] for x in sh])[:n] for k in ["bbox", "bbox_center", "focal_ratio", "rotation", "translation"]}
  R_gt = Rotation.from_quat(np.roll(box["rotation"], -1, -1)).as_matrix()
  t_gt = box["translation"].astype(np.float64)
  ang = lambda X, Y: np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", X, Y) - 1) / 2, -1, 1)))

  def score(Rall, tsc):
    t = recover_translation(tsc, box["bbox"], box["bbox_center"], box["focal_ratio"], TRAN_SCALE, TRAN_OFFSET)
    R = allo_to_ego(Rall, t)
    r = esa(R, t, R_gt, t_gt)
    e = ang(R, R_gt)
    r.update(e_q_gt1=float((e > 1).mean()), e_q_gt10=float((e > 10).mean()))
    return r

  res = {"cmp": a.cmp, "n": int(n), "iters": a.iters}
  for name in ["from_gt (diagnostic)", "from_gt+1-3deg (diagnostic)", "from_gt+3-10deg (diagnostic)"]:
    d = ang(out[name][0], obs.R[:n])
    d0 = ang(starts[name][0], obs.R[:n])
    res[name] = {"start_median_deg": float(np.median(d0)), "final_median_deg": float(np.median(d)), "final_mean_deg": float(d.mean()),
                 "final_gt1": float((d > 1).mean())}
    print(name, json.dumps(res[name]), flush=True)
  res["consensus"] = score(R0, t0)
  res["consensus+comparator"] = score(*out["from_consensus"])
  for k in ["consensus", "consensus+comparator"]:
    print(k, json.dumps(res[k]), flush=True)
  b, c, r = res["from_gt (diagnostic)"], res["consensus"], res["consensus+comparator"]
  res["gate"] = {"bias_ok": b["final_median_deg"] <= 0.3, "esa_ok": r["score"] < c["score"],
                 "median_ok": r["e_q_deg_median"] < c["e_q_deg_median"],
                 "converges_ok": res["from_gt+1-3deg (diagnostic)"]["final_median_deg"] <= 0.5}
  res["gate"]["pass"] = all(res["gate"].values())
  print("gate", json.dumps(res["gate"]), flush=True)
  json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
