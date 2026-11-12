"""Render-and-compare frames for S05: the decimated CAD mesh of object 5 projected with this crop's camera at poses on the
geodesic from the wrong pose (40 deg off) to the GT pose; residual = silhouette XOR the real visible mask."""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/kavinder/ARNABI_ARSH/LARp/video/scenes")
import larc_math as lm
D = "/media/kavinder/hdd3/larc_video/data"
ry, tk, me = np.load(f"{D}/rays.npz"), np.load(f"{D}/tokens.npz"), np.load(f"{D}/mesh5.npz")
k = tk["k"]; V, F = me["v"], me["f"]
mask = np.array(Image.open(f"{D}/mask.png")) > 127
Rw, Rg, t = ry["wr_R_cam"], ry["gt_R_cam"], ry["gt_t_cam"]
w = lm.so3_log(Rw.T @ Rg)
os.makedirs(f"{D}/rc", exist_ok=True)
N = 40
res = []
for i, a in enumerate(np.linspace(0, 1, N)):
  R = Rw @ lm.so3_exp(a * w)
  P = V @ R.T + t
  u = P[:, :2] / P[:, 2:] * k[6:8] + k[8:10]
  uv = (u - k[1:3]) * k[0]
  S = 4  # supersample
  im = Image.new("L", (224 * S, 224 * S), 0)
  dr = ImageDraw.Draw(im)
  for f in F:
    dr.polygon([tuple(uv[j] * S) for j in f], fill=255)
  sil = np.array(im.resize((224, 224), Image.BILINEAR)) > 127
  r = sil ^ mask
  res.append(r.mean())
  ov = np.zeros((224, 224, 4), np.uint8)
  ov[sil] = (255, 183, 77, 60)            # rendering (orange, translucent)
  ov[r] = (240, 98, 146, 235)             # residual (pink)
  Image.fromarray(ov).save(f"{D}/rc/rc_{i:02d}.png")
np.save(f"{D}/rc/residual_frac.npy", np.array(res))
print("residual fraction first/last", res[0], res[-1])
