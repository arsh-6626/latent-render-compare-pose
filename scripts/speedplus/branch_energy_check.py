"""diagnostic (GT used): energy at the GT pose vs at the K=1 samples and at uniform random poses, val_aug first 64 crops"""
import os, sys, glob, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "scripts/speedplus")
from liepose.utils.config import get_args, load_config_from_args
from liepose.exp.cmp import sp_field_dit
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.sp_geometry import crop_meta
from liepose.exp.cmp.sp_field_dit import FtFieldDenoiser
L = os.environ["OUTD"]
sys.argv = [sys.argv[0], "--root_path", L + "/_tb", "den_mode=[field_full_lat_ft]", f"field_dir=[{os.environ['FIELD']}]",
            "train_steps=[100000]", "batch_size=[64]", "n_slices=[64]", "skip_train=[True]", f"restore_path=[{os.environ['RESTORE']}]", "restore_step=[100000]"]
opt = next(iter(load_config_from_args(get_args(exp_name="b", desc="b"), sp_field_dit.build_explorations()).expand())); opt.pop("sample_options", None)
tb = sp_field_dit.Testbed(opt, L + "/_tb", setup=True)
params = {"params": tb.state.params_ema}
D = os.environ["DATADIR"]; obs = TokenSplit(D + "/speedplus_tokens", "val_aug"); meta, _, _ = crop_meta(D + "/speedplus_crops/val_aug")
ims = np.concatenate([np.load(f)["image"] for f in sorted(glob.glob(D + "/speedplus_crops/val_aug/shard_*.npz"))])
n = 64; r = np.arange(n)
z, _ = tb.model.apply(params, jnp.asarray(ims[r], jnp.float32), method=FtFieldDenoiser.encode)
w = jnp.asarray(FG_FLOOR + obs.fg[r]); k = jnp.asarray(meta[r])
def E(R, t):
  zh, _, _ = tb._render(jnp.asarray(R, jnp.float32), jnp.asarray(t, jnp.float32), k, z)
  return np.asarray((w * ((zh - z) ** 2).mean(-1)).sum(-1) / w.sum(-1))
zc = np.load(D + "/speedplus_tokens/val_aug_tokens.npy", mmap_mode="r")[:n]
print("frozen-encode vs cached tokens max abs diff", float(np.abs(np.asarray(z) - zc).max()))
Eg = E(obs.R[r], obs.t[r])
from scipy.spatial.transform import Rotation
rng = np.random.default_rng(0)
Er = np.stack([E(Rotation.random(n, random_state=i).as_matrix(), obs.t[r] + rng.normal(0, 0.3, (n, 3))) for i in range(8)])
for deg in (5, 20, 60, 120, 180):
  Rp = np.stack([Rotation.from_rotvec(Rotation.random(random_state=1000 + j).as_rotvec() / np.linalg.norm(Rotation.random(random_state=1000 + j).as_rotvec()) * np.radians(deg)).as_matrix() @ obs.R[j] for j in r])
  print(f"P(E_gt < E_{deg}deg) = {(Eg < E(Rp, obs.t[r])).mean():.3f}")
print("E_gt mean", Eg.mean(), " E_random mean", Er.mean(), " P(E_gt < E_random)", (Eg[None] < Er).mean())
