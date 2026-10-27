"""Can frozen DINOv2 tokens rank 1-3 deg rotations at all, for a world model that models part of the appearance change?
(SPEED+ synthetic validation crops; torch env, e.g. `jepa`; GT-free probe of the representation, no pose labels used.)

With x a clean val crop, x_aug its colour-augmented copy (val_aug, crops with color_aug), every image resampled at the
same base angle (blur-matched, see tok_sens_inplane.py) and theta an exact in-plane rotation:
  a = z(x_aug) - z(x)          appearance change          d = z(rot_theta x) - z(x)     pose change
A world model that renders the true pose with residual appearance error alpha * a (alpha = 1: ignores appearance, the
tok_sens_inplane.py oracle; alpha = 0.4: the wm2 gate's GT error) sees the observation z(x) + alpha * a; the true pose
wins over the rotated one iff |d|^2 > 2 alpha <d, a>. Reported P(win) per theta and alpha, with the energy's weights
(FG_FLOOR + fg of the observation, as in wm2 / sampling) and unweighted, for 8x8 (2x2-pooled, the cache) and 16x16 tokens.
Optimistic in that the residual is exactly along a (a world model's own error is not); in-plane rotations only.

  python scripts/speedplus/tok_alpha_oracle.py --data $DATADIR --out <json>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok_sens_inplane import rotate  # noqa: E402

THETAS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
ALPHAS = [0.0, 0.25, 0.4, 0.5, 0.75, 1.0]
FG_FLOOR = 0.05  # liepose.exp.jepa_probe.probe_frozen.FG_FLOOR (that module needs jax)


def hull_fg(d, on):
  """(liepose-free: this script runs in the torch env) keypoint hull at the GT pose, crop = (u - (c - dzi/2)) 224/dzi"""
  import cv2
  import scipy.io as sio
  from scipy.ndimage import binary_dilation
  from scipy.spatial.transform import Rotation as Rot
  root = os.environ["SPEEDPLUS_ROOT"]
  cam = json.load(open(os.path.join(root, "camera.json")))
  K, Dd = np.asarray(cam["cameraMatrix"], np.float64), np.asarray(cam["distCoeffs"], np.float64)
  kp = sio.loadmat(os.path.join(root, "tangoPoints.mat"))["tango3Dpoints"].T.astype(np.float64)
  out = []
  for j in np.nonzero(on)[0]:
    R = Rot.from_quat(np.roll(d["rotation"][j], -1)).as_matrix()
    uv, _ = cv2.projectPoints(kp, cv2.Rodrigues(R)[0], d["translation"][j].astype(np.float64), K, Dd)
    dzi = float(d["dzi_size"][j])
    pc = (uv[:, 0] - (d["bbox_center"][j] - dzi / 2)) * 224 / dzi
    m = np.zeros((224, 224), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pc.astype(np.int32)), 1)
    f = cv2.resize(m.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA) > 0.5
    out.append(binary_dilation(f).ravel().astype(np.float32))
  return np.stack(out)


@torch.no_grad()
def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--n", type=int, default=2000)
  p.add_argument("--bs", type=int, default=100)
  p.add_argument("--base", type=float, default=30.0)
  p.add_argument("--hull", action="store_true", help="weights: the object's keypoint hull at the GT pose (8x8 coverage > 0.5, "
                 "dilated by one token; diagnostic: at test time the mask would be the candidate's)")
  a = p.parse_args()
  tok_index = np.load(os.path.join(a.data, "speedplus_tokens", "val_meta.npz"))["index"]
  fg_all = np.load(os.path.join(a.data, "speedplus_tokens", "val_fg.npy")).astype(np.float32)
  pos = {int(i): r for r, i in enumerate(tok_index)}
  xs, xa, fg = [], [], []
  for sc, sa in zip(sorted(glob.glob(os.path.join(a.data, "speedplus_crops", "val", "shard_*.npz"))),
                    sorted(glob.glob(os.path.join(a.data, "speedplus_crops", "val_aug", "shard_*.npz")))):
    dc, da = np.load(sc), np.load(sa)
    assert (dc["index"] == da["index"]).all()
    on = da["color_aug"]
    xs.append(dc["image"][on])
    xa.append(da["image"][on])
    fg.append(hull_fg(dc, on) if a.hull else fg_all[[pos[int(i)] for i in dc["index"][on]]])
    if sum(len(x) for x in xs) >= a.n:
      break
  xs, xa, fg = np.concatenate(xs)[: a.n], np.concatenate(xa)[: a.n], np.concatenate(fg)[: a.n]
  net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
  mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

  def reps(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
      t = net.forward_features((x - mean) / std)["x_norm_patchtokens"].float()
    b = t.shape[0]
    return {"tokens_16x16": t, "tokens_8x8": t.view(b, 8, 2, 8, 2, 384).mean(dim=(2, 4)).reshape(b, 64, 384)}

  S = {}
  for i in range(0, len(xs), a.bs):
    x = torch.from_numpy(xs[i : i + a.bs]).cuda().permute(0, 3, 1, 2).float() / 255
    y = torch.from_numpy(xa[i : i + a.bs]).cuda().permute(0, 3, 1, 2).float() / 255
    w8 = FG_FLOOR + torch.from_numpy(fg[i : i + a.bs]).cuda()
    w16 = w8.view(-1, 8, 1, 8, 1).expand(-1, 8, 2, 8, 2).reshape(-1, 256)
    r0, ra = reps(rotate(x, a.base)), reps(rotate(y, a.base))
    for th in THETAS:
      rt = reps(rotate(x, a.base + th))
      for k in r0:
        dd, aa = rt[k] - r0[k], ra[k] - r0[k]
        for wn, w in [("fg", w8 if k == "tokens_8x8" else w16), ("unw", torch.ones_like(dd[..., 0]))]:
          wm = lambda v: (w * v.mean(-1)).sum(-1) / w.sum(-1)
          S.setdefault((k, wn, th, "d2"), []).append(wm(dd * dd).cpu())
          S.setdefault((k, wn, th, "da"), []).append(wm(dd * aa).cpu())
  S = {k: torch.cat(v).numpy() for k, v in S.items()}
  res = {"n": int(len(xs)), "base_deg": a.base, "thetas_deg": THETAS, "alphas": ALPHAS}
  for k in ["tokens_8x8", "tokens_16x16"]:
    for wn in ["fg", "unw"]:
      tab = {f"{th:g}": {f"{al:g}": round(float((S[(k, wn, th, "d2")] > 2 * al * S[(k, wn, th, "da")]).mean()), 4) for al in ALPHAS}
             for th in THETAS}
      res[f"{k}/{wn}"] = tab
      print(k, wn, json.dumps(tab), flush=True)
  json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
