"""Latent render-and-compare SE(3) diffusion with the v7 world model (probe_ret: crops, retrieval, geometry).

dit.py's token denoiser, with the world model upgraded from the v4 probe (full image, one canonical reference,
ceiling pick 1.61 deg) to v7c (object-centred crops, retrieved references, geometry tokens, ceiling 0.68 deg):

  observation -> crop around its foreground box -> frozen DINOv2 -> z_obs (64 tokens), crop k_obs = (s, x0, y0)
  every denoising step, every sample at its noisy pose g_t:
    reference  = nearest training image to g_t (RefBank, allocentric rotation + symmetric equivalents; in training
                 the observation's own image is excluded)
    render     z_hat = WM(z_ref, g_ref -> g_t), predicted in the observation's crop        (frozen)
    compare    tokens [z_obs, z_hat, z_hat - z_obs, w] -> adaLN transformer conditioned on (t, g_t, k_obs) -> noise

mode = "dit"       as above.
mode = "dit_nowm"  control: z_hat zeroed (a DINOv2-crop-token LiePose denoiser without a world model).
mode = "dit_ray"   dit + where the hypothesis sits relative to the crop: the noisy translation projected into the
                   observation's crop (offset from the crop centre in crop half-sizes) and its depth against the zoom,
                   Fourier-embedded. The SE(3) log rt couples translation to rotation, and the crop otherwise
                   enters only as 3 numbers, so the denoiser never sees the image-space offset it has to correct.

The crop conditioning k_obs (zoom + crop centre) is given to both modes: crop tokens no longer show where the
object is or how large it is, which the translation needs. LiePose's noise schedule, sampler, chordal loss and EMA
are unchanged. The reference lookup is a CPU KD-tree, so sampling is a Python loop over the denoising steps.

  python -m liepose.exp.cmp.run --root_path $LOGS/experiments/symsolt-lrc "den_mode=[dit]" "wm_kind=[ret]" \
      "wm_dir=[$LOGS/probe/v7c_crop_geo_roll]" "cache_dir=[$DATADIR/crop_cache]" "batch_size=[32]" "n_slices=[16]"
"""

import json
import os
from types import SimpleNamespace
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, CachedSplit, weighted_energy
from liepose.exp.jepa_probe.probe_ret import CAM_C, CAM_F, CAM_Z, RefBank, RetPoseModel
from liepose.exp.model import PosEmbed

from .dit import TokenTransformer
from .testbed import Testbed as CmpTestbed

MODES = ("dit", "dit_nowm", "dit_ray")


def crop_feat(k):
  """crop (s, x0, y0) -> log zoom, crop centre in normalised image coordinates"""
  s = k[..., :1]
  return jnp.concatenate([jnp.log(s), (k[..., 1:] + 112.0 / s - 112.0) / 112.0], -1)


def ray_feat(tg, k):
  """hypothesis translation vs crop (s, x0, y0): projected centre offset from the crop centre in crop half-sizes
  (scaled x100, ~0.3 per 0.01 of translation), and log depth minus log zoom (x1000, ~1 per 0.01)"""
  s = k[..., :1]
  Z = jnp.maximum(tg[..., 2:] + CAM_Z, 1.0)  # far-noised hypotheses can leave the frustum
  u = CAM_F * tg[..., :2] / Z + CAM_C
  off = (u - (k[..., 1:] + 112.0 / s)) * s / 112.0
  return jnp.concatenate([100.0 * off, 1000.0 * (jnp.log(Z / CAM_Z) - jnp.log(s / 1.8))], -1)


class RetDenoiser(nn.Module):
  mode: str = "dit"
  dim: int = 256
  depth: int = 4
  heads: int = 8
  wm_dim: int = 384
  wm_depth: int = 6
  wm_geo: bool = True
  tok_dim: int = 384
  geo_fn: Any = None  # world-model geometry for other cameras (sp_geometry.SpeedPlusGeometry)
  crop_fn: Any = None  # crop meta -> conditioning features (default: SYMSOL-T crop_feat)

  def setup(self):
    self.wm = RetPoseModel(dim=self.wm_dim, depth=self.wm_depth, heads=self.wm_dim // 64, out_dim=self.tok_dim,
                           geo=self.wm_geo, geo_fn=self.geo_fn)
    self.net = TokenTransformer(self.dim, self.depth, self.heads)
    self.t_emb = PosEmbed(256, self.dim, True)
    self.g_emb = PosEmbed(256, self.dim, False)
    self.k_emb = nn.Dense(self.dim)
    if self.mode == "dit_ray":
      self.h_emb = PosEmbed(256, self.dim, False)
    self.c_mlp = nn.Dense(self.dim)
    self.noise_out = nn.Dense(6)

  def render(self, zr, Rr, tr, kr, Rg, tg, kg):
    return self.wm(zr, Rr, tr, Rg, tg, kr, kg)

  def noise(self, rt, Rg, tg, t, zr, Rr, tr, kr, z_obs, w, kg):
    z_hat = jax.lax.stop_gradient(self.render(zr, Rr, tr, kr, Rg, tg, kg))
    if self.mode == "dit_nowm":
      z_hat = jnp.zeros_like(z_hat)
    h = self.t_emb(t) + self.g_emb(rt) + self.k_emb((self.crop_fn or crop_feat)(kg))
    if self.mode == "dit_ray":
      h = h + self.h_emb(ray_feat(tg, kg))
    c = self.c_mlp(jax.nn.silu(h))
    return self.noise_out(self.net(z_obs, z_hat, w, c))

  def __call__(self, *args):
    return self.noise(*args)


class SplitCache:
  """Crop tokens of one split (crop_encode.py split), indexed like the SymsoltDataset (annotation order)."""

  def __init__(self, tok, fg, label, crop):
    self.tok, self.fg, self.label, self.crop = tok, fg, label, crop

  def gather(self, index, label):
    index = np.asarray(index)
    assert (self.label[index] == np.asarray(label)).all(), "token cache misaligned with dataset index"
    return self.tok[index], self.fg[index]

  def extra(self, index):
    return (self.crop[np.asarray(index)],)


class Testbed(CmpTestbed):

  def __init__(self, a, exp_path, setup=True):
    a.cond_mode = "none"  # cmp conditioning path unused
    self.den_mode = a.den_mode
    assert self.den_mode in MODES, self.den_mode
    super().__init__(a, exp_path, setup=setup)

  # --- world model, reference bank, caches ---

  def create_model(self):
    wa = SimpleNamespace(**json.load(open(os.path.join(self.a.wm_dir, "args.json"))))
    assert wa.crop and wa.pool == 2 and not getattr(wa, "resid", False), "expects a probe_ret crop model"
    self.wa = wa
    self.train_split = CachedSplit(self.a.cache_dir, wa.train_split, wa.pool)
    self.train_crop = np.load(os.path.join(self.a.cache_dir, f"{wa.train_split}_p{wa.pool}_meta.npz"))["crop"].astype(np.float32)
    self.bank = RefBank(self.train_split, wa.lam, not wa.no_sym)
    model = RetDenoiser(mode=self.den_mode, dim=self.a.dit_dim, depth=self.a.dit_depth, wm_dim=wa.dim, wm_depth=wa.depth,
                        wm_geo=wa.geo)
    n, z, e, k = 2, jnp.zeros((2, 64, 384)), jnp.tile(jnp.eye(3), (2, 1, 1)), jnp.tile(jnp.array([[2.0, 50.0, 50.0]]), (2, 1))
    v3 = jnp.zeros((n, 3))
    params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 6)), e, v3, jnp.zeros((n, 1)), z, e, v3, k,
                                 z, jnp.ones((n, 64)), k))
    src = flax.serialization.msgpack_restore(open(os.path.join(self.a.wm_dir, "params.msgpack"), "rb").read())["params"]
    shapes = lambda p: jax.tree_util.tree_map(np.shape, p)
    assert shapes(src) == shapes(params["params"]["wm"]), "world model checkpoint does not match RetPoseModel"
    params["params"]["wm"] = jax.tree_util.tree_map(jnp.asarray, src)
    params["batch_stats"] = {}
    return model, FrozenDict(params)

  def create_cond_fn(self):
    return None

  def cache(self, split):
    if split not in self.caches:
      if split == "train":
        assert self.a.cache_train == self.wa.train_split
        ts = self.train_split
        self.caches[split] = SplitCache(ts.tokens, ts.fg, ts.label, self.train_crop)
      else:
        d, name = self.a.cache_dir, self.a.cache_test
        meta = np.load(os.path.join(d, f"{name}_p2_meta.npz"))
        self.caches[split] = SplitCache(np.load(os.path.join(d, f"{name}_p2_tokens.npy")),
                                        np.load(os.path.join(d, f"{name}_p2_fg.npy")).astype(np.float32), meta["label"],
                                        meta["crop"].astype(np.float32))
    return self.caches[split]

  def _poses_fn(self):
    lm = self.lie_metrics

    @jax.jit
    def poses(rt):
      g = jax.vmap(lm.as_lie)(rt)
      return jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g)

    return poses

  def _refs(self, shape, R, t, exclude):
    """host-side retrieval at the (noisy) poses -> reference tokens, pose, crop"""
    img, Rr, tr = self.bank.query(shape, np.asarray(R, np.float64), np.asarray(t, np.float64), exclude)
    return self.train_split.tokens[img], Rr.astype(np.float32), tr.astype(np.float32), self.train_crop[img]

  # --- train ---

  def create_get_flat_batch_train_fn(self):
    base = super().create_get_flat_batch_train_fn()  # + tok, fg, shape (per image)
    poses = self._poses_fn()

    def get_flat_batch_train(seed, batch, n_slices=256):
      b = base(seed, batch, n_slices=n_slices)
      index = np.asarray(batch["index"])
      s = b["rt"].shape[0] // len(index)
      Rg, tg = map(np.asarray, poses(b["rt"]))
      zr, Rr, tr, kr = self._refs(np.repeat(np.asarray(batch["label_shape"]), s), Rg, tg, np.repeat(index, s))
      b.update(Rg=jnp.asarray(Rg), tg=jnp.asarray(tg), zr=jnp.asarray(zr), Rr=jnp.asarray(Rr), tr=jnp.asarray(tr),
               kr=jnp.asarray(kr), kg=jnp.asarray(np.repeat(self.train_crop[index], s, 0)))
      b.pop("img", None)
      return b

    return get_flat_batch_train

  def create_update_fn(self):
    lie_metrics, loss_name, model = self.lie_metrics, self.a.loss_name, self.model
    assert self.a.learn_noise

    def update_fn(state, batch):
      def loss_fn(params):
        t = batch["t"].reshape((-1, 1)).astype(jnp.float32)
        s = batch["rt"].shape[0] // batch["tok"].shape[0]
        z_obs = jnp.repeat(batch["tok"].astype(jnp.float32), s, 0)
        w = jnp.repeat(FG_FLOOR + batch["fg"], s, 0)
        mu = model.apply({"params": params}, batch["rt"], batch["Rg"], batch["tg"], t, batch["zr"].astype(jnp.float32),
                         batch["Rr"], batch["tr"], batch["kr"], z_obs, w, batch["kg"], method=RetDenoiser.noise)
        return jnp.mean(jax.vmap(lie_metrics.get_distance_fn(loss_name))(batch["zt"], mu))

      loss, grads = jax.value_and_grad(loss_fn)(state.params)
      new_state = state.apply_gradients(grads=grads)
      # world model frozen (restore it: AdamW weight decay would still move it)
      new_state = new_state.replace(params=freeze({**unfreeze(new_state.params), "wm": state.params["wm"]}))
      return loss, new_state

    return update_fn

  # --- sampling ---

  def _params(self):
    return {"params": self.state.params_ema if self.a.ema_enabled else self.state.params}

  def create_energy_fn(self, t=0):
    """World-model energy of final samples (reference retrieved at each sample's pose; lower = better)."""
    model, poses = self.model, self._poses_fn()
    params = self._params()

    @jax.jit
    def energy_j(zr, Rr, tr, kr, Rg, tg, kg, z_obs, w):
      z_hat = model.apply(params, zr.astype(jnp.float32), Rr, tr, kr, Rg, tg, kg, method=RetDenoiser.render)
      return weighted_energy(z_hat, z_obs, w)

    def energy(rt, shape, tok, fg, crop):
      s = rt.shape[0] // tok.shape[0]
      Rg, tg = map(np.asarray, poses(rt))
      zr, Rr, tr, kr = self._refs(np.repeat(np.asarray(shape), s), Rg, tg, np.full(len(Rg), -1))
      return energy_j(zr, Rr, tr, kr, Rg, tg, jnp.repeat(crop, s, 0), jnp.repeat(tok, s, 0), jnp.repeat(FG_FLOOR + fg, s, 0))

    return energy

  def create_sampler(self, n_steps=100):
    model, poses = self.model, self._poses_fn()
    params = self._params()
    p_fn = jax.vmap(self.create_p_sample_fn())
    T = self.noise_schedule.timesteps
    cur = (np.linspace(T, 0, n_steps, endpoint=False) - 1).astype(np.int32).tolist()
    seq = list(zip(cur, cur[1:] + [0]))

    @jax.jit
    def step(key, rt, t, tp, Rg, tg, zr, Rr, tr, kr, z_obs, w, kg):
      B = rt.shape[0]
      key, k = jax.random.split(key)
      mu = model.apply(params, rt, Rg, tg, jnp.full((B, 1), t, jnp.float32), zr.astype(jnp.float32), Rr, tr, kr, z_obs, w,
                       kg, method=RetDenoiser.noise)
      _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      return key, rt

    def sample(key, img, shape, tok, fg, rt, crop):
      s = rt.shape[0] // tok.shape[0]
      shape_rep = np.repeat(np.asarray(shape), s)
      z_obs, w, kg = jnp.repeat(tok, s, 0), jnp.repeat(FG_FLOOR + fg, s, 0), jnp.repeat(crop, s, 0)
      for t, tp in seq:
        Rg, tg = poses(rt)
        zr, Rr, tr, kr = self._refs(shape_rep, np.asarray(Rg), np.asarray(tg), np.full(len(shape_rep), -1))
        key, rt = step(key, rt, jnp.int32(t), jnp.int32(tp), Rg, tg, zr, Rr, tr, kr, z_obs, w, kg)
      return rt

    return sample
