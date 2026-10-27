"""How much do frozen DINOv2 tokens change under a small, exact pose change, compared with an appearance change?
(SPEED+ synthetic validation crops; torch env, e.g. `jepa`.)

Pose change: in-plane rotation of the crop by theta about its centre (a true rotation about the camera axis, rendered
exactly by resampling). Appearance change: the same crop with LiePose's colour augmentation (val_aug, the crops with
color_aug set). Every image, including the reference, is resampled at a nonzero angle (reference: --base deg), so
bilinear interpolation blur and patch-grid aliasing are common to all of them and do not count as a pose change. The base
must be large: at 0.05 deg the resampling is nearly the identity (no blur) while every rotated image is blurred, which
inflates the token change (8x8 ratio at 1 deg 0.59 at base 0.05 vs 0.24 at base 10 / 30) and pushes the oracle below
chance (blurred renders sit closer to the augmented crop). Default base 30 deg. For each representation (pixels, unpooled 16x16 tokens, 2x2-pooled 8x8 tokens):
  acc(theta)   = P(||z(rot x) - z(x)|| > ||z(x_aug) - z(x)||)     is the pose change larger than the appearance change
  ratio(theta) = median ||z(rot x) - z(x)|| / median ||z(x_aug) - z(x)||
  oracle(theta) = P(||z(rot x) - z(x_aug)|| > ||z(x) - z(x_aug)||)   render-and-compare with a perfect world model that
                  renders the clean appearance exactly: the observation is the augmented crop, the candidate renders are
                  the clean crop at the true pose and rotated by theta. The ceiling of P(E_pert > E_gt) for frozen tokens
                  (in-plane rotations only; the world model's gate is P >= 0.9 at 1-3 deg)
Distances are means over all positions (SPEED+ crops have no masks).

  python scripts/speedplus/tok_sens_inplane.py --clean $DATADIR/speedplus_crops/val --aug $DATADIR/speedplus_crops/val_aug \
      --out <json>
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

THETAS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]


def rotate(x, deg):
  th = math.radians(deg)
  A = torch.tensor([[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0]], device=x.device)
  grid = F.affine_grid(A[None].expand(x.shape[0], 2, 3), x.shape, align_corners=False)
  return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


@torch.no_grad()
def main():
  p = argparse.ArgumentParser()
  p.add_argument("--clean", required=True)
  p.add_argument("--aug", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--n", type=int, default=2000)
  p.add_argument("--bs", type=int, default=100)
  p.add_argument("--base", type=float, default=30.0)
  a = p.parse_args()
  xs, xa = [], []
  for sc, sa in zip(sorted(glob.glob(os.path.join(a.clean, "shard_*.npz"))), sorted(glob.glob(os.path.join(a.aug, "shard_*.npz")))):
    dc, da = np.load(sc), np.load(sa)
    assert (dc["index"] == da["index"]).all()
    on = da["color_aug"]
    xs.append(dc["image"][on])
    xa.append(da["image"][on])
    if sum(len(x) for x in xs) >= a.n:
      break
  xs, xa = np.concatenate(xs)[: a.n], np.concatenate(xa)[: a.n]
  net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
  mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

  def reps(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
      t = net.forward_features((x - mean) / std)["x_norm_patchtokens"].float()
    b = t.shape[0]
    pooled = t.view(b, 8, 2, 8, 2, 384).mean(dim=(2, 4)).reshape(b, 64, 384)
    return {"pixels": x.reshape(b, 3, -1).transpose(1, 2), "tokens_16x16": t, "tokens_8x8": pooled}

  d = lambda u, v: ((u - v) ** 2).mean((1, 2)).sqrt()
  D = {}
  for i in range(0, len(xs), a.bs):
    x = torch.from_numpy(xs[i : i + a.bs]).cuda().permute(0, 3, 1, 2).float() / 255
    y = torch.from_numpy(xa[i : i + a.bs]).cuda().permute(0, 3, 1, 2).float() / 255
    r0, ra = reps(rotate(x, a.base)), reps(rotate(y, a.base))
    for k in r0:
      D.setdefault((k, "app"), []).append(d(ra[k], r0[k]).cpu())
    for th in THETAS:
      rt = reps(rotate(x, a.base + th))
      for k in r0:
        D.setdefault((k, th), []).append(d(rt[k], r0[k]).cpu())
        D.setdefault((k, "obs", th), []).append(d(rt[k], ra[k]).cpu())
  D = {k: torch.cat(v).numpy() for k, v in D.items()}
  res = {"n": int(len(xs)), "thetas_deg": THETAS, "base_deg": a.base}
  for k in ["pixels", "tokens_16x16", "tokens_8x8"]:
    app = D[(k, "app")]
    res[k] = {f"{th:g}": {"acc": round(float((D[(k, th)] > app).mean()), 4),
                          "ratio": round(float(np.median(D[(k, th)]) / np.median(app)), 3),
                          "oracle": round(float((D[(k, "obs", th)] > app).mean()), 4)} for th in THETAS}
    print(k, {t: (v["acc"], v["ratio"], v["oracle"]) for t, v in res[k].items()}, flush=True)
  json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
