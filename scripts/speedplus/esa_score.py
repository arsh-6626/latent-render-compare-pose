"""SPEED+ (ESA Pose Estimation Challenge / SPEC2021) errors for world-model ceiling picks.

Reads energies_<split>.npz written by probe_ret_tless (allocentric rotation + translation_scaled candidates around
each observation) and the matching crop shards (bbox, focal ratio, GT pose), and reports, for the min-energy pick:

  e_q  = 2 arccos |<q, q_gt>|          egocentric rotation, rad (deg also reported)
  e_t  = ||t - t_gt|| / ||t_gt||        camera-frame translation, metres
  score = mean(e_q* + e_t*)             with the challenge's precision floors: e_q* = 0 if e_q < 0.169 deg,
                                        e_t* = 0 if e_t < 2.173 mm/m

The rotation pick is the min-energy rotation candidate (all at the GT translation); the translation pick is the
min-energy translation candidate (all at the GT rotation); the scored pose combines the two. Candidates are drawn
around the GT (and the GT is one of them), so these are refinement-ceiling numbers, not pose estimates; the
"random pick" rows give the chance level on the same candidate sets.

  PYTHONPATH=src python scripts/speedplus/esa_score.py --energies $LOGS/probe/sp_ret_v1 \
      --crops $DATADIR/speedplus_crops --splits val lightbox sunlamp
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from liepose.data.speedplus.dataset import recover_translation
from liepose.exp.speedplus.run import TRAN_OFFSET, TRAN_SCALE
from liepose.utils.npfn import get_allo_rotation

ROT_FLOOR = np.radians(0.169)
TRAN_FLOOR = 2.173e-3


def load_crops(path):
  keys = ["index", "rotation", "rotation_allo", "translation", "translation_scaled", "bbox", "bbox_center", "focal_ratio"]
  sh = [np.load(f) for f in sorted(glob.glob(os.path.join(path, "shard_*.npz")))]
  return {k: np.concatenate([s[k] for s in sh]) for k in keys}


def allo_to_ego(R_allo, t):
  """R_allo = A(t) R_ego (CropResizeAffiner.transform_pose)  ->  R_ego"""
  A = Rotation.from_quat(np.roll(np.stack([get_allo_rotation(x) for x in t]), -1, -1))  # wxyz -> xyzw
  return (A.inv() * Rotation.from_matrix(R_allo)).as_matrix()


def esa(R, t, R_gt, t_gt):
  e_q = np.arccos(np.clip((np.einsum("nij,nij->n", R, R_gt) - 1) / 2, -1, 1))  # = 2 arccos|<q,q_gt>|
  e_t = np.linalg.norm(t - t_gt, axis=-1) / np.linalg.norm(t_gt, axis=-1)
  s_q, s_t = np.where(e_q < ROT_FLOOR, 0, e_q), np.where(e_t < TRAN_FLOOR, 0, e_t)
  return {"score": float((s_q + s_t).mean()), "score_ori": float(s_q.mean()), "score_pos": float(s_t.mean()),
          "e_q_deg_mean": float(np.degrees(e_q).mean()), "e_q_deg_median": float(np.degrees(np.median(e_q))),
          "e_t_m_mean": float(np.linalg.norm(t - t_gt, axis=-1).mean()),
          "e_t_m_median": float(np.median(np.linalg.norm(t - t_gt, axis=-1)))}


def score_split(E_npz, crops, rng):
  d = np.load(E_npz)
  rows, Rc, tc, E, kind, is_rot = d["rows"], d["R"], d["t"], d["E"], d["kind"], d["is_rot"]
  c = {k: v[rows] for k, v in crops.items()}
  # GT allocentric pose: the candidates' GT slot must be the crop's GT
  assert np.abs(tc[:, 0] - c["translation_scaled"]).max() < 1e-4
  t_gt = c["translation"].astype(np.float64)
  R_gt = Rotation.from_quat(np.roll(c["rotation"], -1, -1)).as_matrix()
  rec = lambda ts: recover_translation(ts, c["bbox"], c["bbox_center"], c["focal_ratio"], TRAN_SCALE, TRAN_OFFSET)
  t_rec = rec(tc[:, 0].astype(np.float64))
  R_ego_check = allo_to_ego(Rc[:, 0].astype(np.float64), t_gt)
  sanity = {"recover_t_err_m_max": float(np.linalg.norm(t_rec - t_gt, axis=-1).max()),
            "allo_to_ego_err_deg_max": float(np.degrees(esa_angle(R_ego_check, R_gt).max()))}

  n = len(rows)
  tran = kind == "tran"
  out = {"n_obs": int(n), "sanity": sanity}
  for name, pick in [("min_energy", lambda m: np.argmin(np.where(m[None], E, np.inf), 1)),
                     ("random_pick", lambda m: np.array([rng.choice(np.nonzero(m)[0]) for _ in range(n)]))]:
    rot_m = is_rot & (kind != "uni")  # GT + the 1-30 deg perturbations
    for cand, m in [("near", rot_m), ("all", is_rot)]:
      ir, it = pick(m), pick(tran | (kind == "gt"))
      R_allo = Rc[np.arange(n), ir].astype(np.float64)
      t_m = rec(tc[np.arange(n), it].astype(np.float64))
      out[f"{name}/{cand}"] = esa(allo_to_ego(R_allo, t_m), t_m, R_gt, t_gt)
  return out


def esa_angle(R, R_gt):
  return np.arccos(np.clip((np.einsum("nij,nij->n", R, R_gt) - 1) / 2, -1, 1))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--energies", nargs="+", required=True, help="probe dirs holding energies_<split>.npz")
  p.add_argument("--crops", required=True)
  p.add_argument("--splits", nargs="+", default=["val", "lightbox", "sunlamp"])
  a = p.parse_args()
  crops = {s: load_crops(os.path.join(a.crops, s)) for s in a.splits}
  for e in a.energies:
    res = {s: score_split(os.path.join(e, f"energies_{s}.npz"), crops[s], np.random.default_rng(0)) for s in a.splits}
    json.dump(res, open(os.path.join(e, "esa_score.json"), "w"), indent=2)
    print(f"== {e}")
    for s, r in res.items():
      print(f"  {s}  sanity {r['sanity']}")
      for k, v in r.items():
        if k not in ("n_obs", "sanity"):
          print(f"    {k:22s} score {v['score']:.4f}  ori {v['score_ori']:.4f}  pos {v['score_pos']:.4f}  "
                f"e_q {v['e_q_deg_mean']:.2f}/{v['e_q_deg_median']:.2f} deg  e_t {v['e_t_m_mean']:.3f}/{v['e_t_m_median']:.3f} m")


if __name__ == "__main__":
  main()
