"""Refinement ceiling, step 2: can a feature space resolve fine pose errors, given an exact renderer?
(torch-CUDA env, e.g. `jepa`)

Distance d(render(g), observation) in: raw pixels, DINOv2 16x16 patch tokens, 2x2-pooled 8x8 tokens (what
our world model predicts), CLS token. Patch distances use the world-model weighting (FG_FLOOR + foreground).
Reports, per feature space:
  - discrimination: P(d(perturbed) > d(GT)) by symmetry-aware rotation error bin, and by translation offset
  - selection: error of argmin_d over the perturbed rotation candidates (GT excluded) vs. random and best
    candidate, i.e. how precisely a refiner that trusts this distance could land, starting 0.25-8 deg away.

  python scripts/oracle/ceiling_feats.py --ceiling $DATADIR/oracle/ceiling.npz --out $DATADIR/oracle/ceiling_feats.json
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

FG_FLOOR = 0.05
ROT_BINS = [(0.1, 0.35), (0.35, 0.75), (0.75, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 10.0)]


def fg_weights(images):
  m = torch.from_numpy((images.max(axis=-1) > 10).astype(np.float32))[:, None]
  return F.max_pool2d(F.adaptive_avg_pool2d(m, 16), 3, stride=1, padding=1)[:, 0]  # (B,16,16)


@torch.no_grad()
def encode(net, images):
  mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
  x = torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float() / 255
  with torch.autocast("cuda", dtype=torch.bfloat16):
    o = net.forward_features((x - mean) / std)
  return o["x_norm_patchtokens"].float(), o["x_norm_clstoken"].float()


def distances(net, obs, renders):
  """obs (H,W,3), renders (C,H,W,3) uint8 -> dict name -> (C,) distances to obs."""
  imgs = np.concatenate([obs[None], renders])
  tok, cls = encode(net, imgs)  # (1+C,256,384)
  w16 = (FG_FLOOR + fg_weights(obs[None])[0]).cuda().reshape(256)
  w8 = (FG_FLOOR + F.max_pool2d(fg_weights(obs[None]), 2)[0]).cuda().reshape(64)
  t8 = tok.view(-1, 8, 2, 8, 2, 384).mean(dim=(2, 4)).reshape(-1, 64, 384)
  out = {}
  out["pixel"] = ((renders.astype(np.float32) - obs.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
  e16 = ((tok[1:] - tok[:1]) ** 2).mean(-1)
  out["dino16"] = ((e16 * w16).sum(-1) / w16.sum()).cpu().numpy()
  e8 = ((t8[1:] - t8[:1]) ** 2).mean(-1)
  out["dino8_pooled"] = ((e8 * w8).sum(-1) / w8.sum()).cpu().numpy()
  out["cls"] = ((cls[1:] - cls[:1]) ** 2).mean(-1).cpu().numpy()
  return out


def summarize(D, rot_err, tran_err, kind, label, shapes):
  """D (N,C) distances. GT is candidate 0."""
  N = len(D)
  res = {}
  gt = D[:, :1]
  rot = kind == "rot"
  disc = {}
  for lo, hi in ROT_BINS:
    m = rot & (rot_err >= lo) & (rot_err < hi)
    disc[f"{lo}-{hi}deg"] = float((D[m] > np.broadcast_to(gt, D.shape)[m]).mean()) if m.any() else None
  res["P(d_pert>d_gt) by rot err"] = disc
  tdisc = {}
  tr = kind == "tran"
  for lv in np.unique(tran_err[tr]):
    m = tr & np.isclose(tran_err, lv)
    tdisc[f"{lv:g}"] = float((D[m] > np.broadcast_to(gt, D.shape)[m]).mean())
  res["P(d_pert>d_gt) by tran offset"] = tdisc
  # selection among perturbed rotation candidates (GT excluded)
  sel, best, rnd = [], [], []
  for i in range(N):
    c = np.nonzero(rot[i])[0]
    sel.append(rot_err[i, c[np.argmin(D[i, c])]])
    best.append(rot_err[i, c].min())
    rnd.append(rot_err[i, c].mean())
  sel, best, rnd = map(np.asarray, (sel, best, rnd))
  res["select_rot_err"] = {"argmin_d": float(sel.mean()), "best_candidate": float(best.mean()),
                           "random": float(rnd.mean()),
                           "per_shape_argmin_d": {s: float(sel[label == k].mean()) for k, s in enumerate(shapes)}}
  return res


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ceiling", required=True)
  p.add_argument("--out", required=True)
  a = p.parse_args()
  C = np.load(a.ceiling)
  net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
  N = len(C["index"])
  D = {}
  renders = C["renders"]
  obs = C["obs"]
  for i in range(N):
    d = distances(net, obs[i], renders[i])
    for k, v in d.items():
      D.setdefault(k, []).append(v)
  D = {k: np.stack(v) for k, v in D.items()}
  np.savez(a.out.replace(".json", "_dist.npz"), **D)
  shapes = ["tet", "cube", "icosa", "cone", "cyl"]
  res = {k: summarize(v, C["rot_err"], C["tran_err"], C["kind"], C["label"], shapes) for k, v in D.items()}
  json.dump(res, open(a.out, "w"), indent=2)
  for k, r in res.items():
    print(f"\n== {k}")
    print("  P(d_pert > d_gt) by sym-aware rot err: " + "  ".join(f"{b}: {v:.3f}" for b, v in r["P(d_pert>d_gt) by rot err"].items() if v is not None))
    print("  P(d_pert > d_gt) by translation offset: " + "  ".join(f"{b}: {v:.3f}" for b, v in r["P(d_pert>d_gt) by tran offset"].items()))
    s = r["select_rot_err"]
    print(f"  pick among perturbed (0.25-8deg): argmin_d {s['argmin_d']:.2f} deg | best {s['best_candidate']:.2f} | "
          f"random {s['random']:.2f} | per shape " + " ".join(f"{k}={v:.2f}" for k, v in s["per_shape_argmin_d"].items()))


if __name__ == "__main__":
  main()
