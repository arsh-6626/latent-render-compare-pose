"""Learned tri-planes of object 5 (PCA to RGB) and the training viewing directions of object 5 (liepose env, CPU, read-only)."""
import flax
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

OUT = "/media/kavinder/hdd3/larc_video/data"
p = flax.serialization.msgpack_restore(open("/media/kavinder/hdd3/larc_video/ckpt/tl_field_v1/params.msgpack", "rb").read())
pp = p["params"]
print(list(pp.keys()))
planes = [np.asarray(pp[f"plane_{n}"])[5 - 1] for n in ("xy", "xz", "yz")]  # (128, 128, 32) each
X = np.concatenate([q.reshape(-1, 32) for q in planes])
mu = X.mean(0)
_, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
Y = (X - mu) @ Vt[:3].T
lo, hi = np.percentile(Y, 1, 0), np.percentile(Y, 99, 0)
for n, q in zip(("xy", "xz", "yz"), planes):
  c = np.clip(((q.reshape(-1, 32) - mu) @ Vt[:3].T - lo) / (hi - lo), 0, 1).reshape(128, 128, 3)
  Image.fromarray((c * 255).astype(np.uint8)).save(f"{OUT}/plane_{n}.png")  # row = second coordinate, col = first
m = np.load("/media/kavinder/hdd2/liepose_data/tless_tokens/train_meta.npz")
sel = np.where(m["object_id"] == 5)[0]
q = m["rotation_allo"][sel]
R = Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], -1)).as_matrix()
view = -np.einsum("nji,j->ni", R, np.array([0, 0, 1.0]))  # camera direction in the object frame: R^T (0,0,-1)
rng = np.random.default_rng(0)
np.savez(f"{OUT}/train_views.npz", view=view[rng.choice(len(view), min(1500, len(view)), replace=False)], n=len(sel))
print("train crops of obj 5:", len(sel))
