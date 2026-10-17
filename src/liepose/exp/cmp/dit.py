"""World-model-native denoisers for SYMSOL-T SE(3) diffusion (replace LiePose's ResNet + MLP head).

mode = "dit"       Option 1. A transformer over the 64 DINOv2 patch tokens of the observation. Each token
                   carries [z_obs, ẑ(g_t), ẑ(g_t) - z_obs, mask weight], where ẑ(g_t) = P(z_ref, g_t) is the
                   frozen world model's prediction at the noisy pose; (g_t, t) condition every block (adaLN)
                   and a prepended token, which outputs the noise. Token-level latent render-and-compare.
mode = "dit_nowm"  Control for option 1: same transformer, ẑ zeroed (DINOv2-token denoiser without a world model).
mode = "energy"    Option 2. A scalar energy E(g, x, t) from the same token transformer, where the pose enters
                   ONLY through the world model's prediction ẑ(g) (render) compared with z_obs (compare); the
                   predicted noise is ∇_ξ E(g_t·exp(ξ), x, t)|_{ξ=0}. The world model is fine-tuned jointly.

mode = "fm_dit"    Flow matching instead of score diffusion: rotation follows the SO(3) geodesic and translation a
                   straight line from a prior pose x0 to the GT pose x1, R_s = R0 exp(s log(R0^T R1)),
                   t_s = t0 + s (t1 - t0); the token transformer (with ẑ(x_s)) regresses the constant velocity
                   (body-frame ω, Δt). Sampling: Euler ODE, few steps.
mode = "fm_dit_nowm"  Flow-matching control without the world model.

For the diffusion modes, LiePose's noise schedule, sampler (with the principal-log fix), chordal loss and EMA
are unchanged.
"""

import json
import os

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

from liepose.exp.jepa_probe.probe import AdaLNBlock
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, FrozenPoseModel, pool_tokens
from liepose.exp.model import PosEmbed

from .testbed import Testbed as CmpTestbed
from .testbed import pose_feat_jnp

MODES = ("dit", "dit_nowm", "energy", "fm_dit", "fm_dit_nowm")
FM_MODES = ("fm_dit", "fm_dit_nowm")


class TokenTransformer(nn.Module):
  """Tokens [z_obs, ẑ, ẑ - z_obs, w] + one conditioning token, adaLN blocks; returns the conditioning token."""
  dim: int = 256
  depth: int = 4
  heads: int = 8

  @nn.compact
  def __call__(self, z_obs, z_hat, w, c):
    x = jnp.concatenate([z_obs, z_hat, z_hat - z_obs, w[..., None]], axis=-1)
    x = nn.Dense(self.dim, name="tok_in")(nn.LayerNorm()(x))
    x = x + self.param("pos", nn.initializers.normal(0.02), (1, x.shape[1], self.dim))
    x = jnp.concatenate([c[:, None], x], axis=1)
    for i in range(self.depth):
      x = AdaLNBlock(self.dim, self.heads, name=f"block_{i}")(x, c)
    return nn.LayerNorm()(x[:, 0])


class WMDenoiser(nn.Module):
  mode: str = "dit"
  dim: int = 256
  depth: int = 4
  heads: int = 8
  wm_dim: int = 384
  wm_depth: int = 6
  tok_dim: int = 384

  def setup(self):
    self.wm = FrozenPoseModel(dim=self.wm_dim, depth=self.wm_depth, heads=self.wm_dim // 64, out_dim=self.tok_dim)
    self.net = TokenTransformer(self.dim, self.depth, self.heads)
    self.t_emb = PosEmbed(256, self.dim, True)
    self.g_emb = PosEmbed(256, self.dim, False)
    self.c_mlp = nn.Dense(self.dim)
    self.noise_out = nn.Dense(6)
    self.energy_out = nn.Dense(1)

  def predict_tokens(self, z_ref, g9):
    return self.wm(z_ref, g9)

  def noise(self, rt, g9, t, z_ref, z_obs, w):
    """Options 1 / control: predict the noise directly (flow modes: predict the velocity; rt is then g9)."""
    z_hat = jax.lax.stop_gradient(self.wm(z_ref, g9))
    if self.mode in ("dit_nowm", "fm_dit_nowm"):
      z_hat = jnp.zeros_like(z_hat)
    c = self.c_mlp(jax.nn.silu(self.t_emb(t) + self.g_emb(rt)))
    return self.noise_out(self.net(z_obs, z_hat, w, c))

  def energy(self, g9, t, z_ref, z_obs, w):
    """Option 2: scalar energy; the pose reaches it only through the world model's prediction."""
    z_hat = self.wm(z_ref, g9)
    c = self.c_mlp(jax.nn.silu(self.t_emb(t)))
    return self.energy_out(self.net(z_obs, z_hat, w, c))[:, 0]

  def __call__(self, rt, g9, t, z_ref, z_obs, w):  # used for init: touch every submodule
    return self.noise(rt, g9, t, z_ref, z_obs, w) + self.energy(g9, t, z_ref, z_obs, w)[:, None]


class Testbed(CmpTestbed):
  """Same data/caches/sampler plumbing as cmp.Testbed, with a WM-native denoiser instead of ResNet + head."""

  def __init__(self, a, exp_path, setup=True):
    a.cond_mode = "none"  # cmp conditioning path unused
    self.den_mode = a.den_mode
    assert self.den_mode in MODES, self.den_mode
    super().__init__(a, exp_path, setup=setup)

  def _refs(self):
    refs, _ = pool_tokens(np.load(os.path.join(self.a.cache_dir, "refs_tokens.npy")),
                          np.load(os.path.join(self.a.cache_dir, "refs_fg.npy")), 2)
    return jnp.asarray(refs, jnp.float32)

  def create_model(self):
    with open(os.path.join(self.a.wm_dir, "args.json")) as f:
      pa = json.load(f)
    assert pa["pool"] == 2
    model = WMDenoiser(mode=self.den_mode, dim=self.a.dit_dim, depth=self.a.dit_depth, wm_dim=pa["dim"], wm_depth=pa["depth"])
    z = jnp.zeros((2, 64, 384))
    rt_dim = 9 if self.den_mode in FM_MODES else 6
    params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((2, rt_dim)), jnp.zeros((2, 9)), jnp.zeros((2, 1)),
                                 z, z, jnp.zeros((2, 64))))
    # world-model weights from the trained probe (frozen in "dit", fine-tuned in "energy")
    fm = FrozenPoseModel(dim=pa["dim"], depth=pa["depth"], heads=pa["dim"] // 64, out_dim=384)
    wm_init = fm.init(jax.random.PRNGKey(0), jnp.zeros((1, 64, 384)), jnp.zeros((1, 9)))
    with open(os.path.join(self.a.wm_dir, "params.msgpack"), "rb") as f:
      wm_params = flax.serialization.from_bytes(wm_init, f.read())
    params["params"]["wm"] = jax.tree_util.tree_map(jnp.asarray, unfreeze(wm_params)["params"])
    params["batch_stats"] = {}
    self.refs = self._refs()
    return model, FrozenDict(params)

  def create_cond_fn(self):
    return None

  # --- shared helpers ---

  def _inputs(self, rt, shape, tok, fg):
    s = rt.shape[0] // tok.shape[0]
    return jnp.repeat(self.refs[shape], s, 0), jnp.repeat(tok, s, 0), jnp.repeat(FG_FLOOR + fg, s, 0)

  def _mu_fn(self):
    model, lm, Lie, mode = self.model, self.lie_metrics, self.Lie, self.den_mode

    def g9_of(rt, xi=None):
      g = jax.vmap(lm.as_lie)(rt)
      if xi is not None:
        g = jax.vmap(lambda a, x: a @ Lie.exp(x))(g, xi)
      return pose_feat_jnp(jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g))

    def mu(params, rt, t, z_ref, z_obs, w):
      if mode == "energy":
        def e_sum(xi):
          return model.apply(params, g9_of(rt, xi), t, z_ref, z_obs, w, method=WMDenoiser.energy).sum()
        return jax.grad(e_sum)(jnp.zeros_like(rt))
      return model.apply(params, rt, g9_of(rt), t, z_ref, z_obs, w, method=WMDenoiser.noise)

    return mu

  # --- flow matching on SO(3) x R3 ---

  def create_get_flat_batch_train_fn(self):
    base = super().create_get_flat_batch_train_fn()
    if self.den_mode not in FM_MODES:
      return base
    lm, Lie = self.lie_metrics, self.Lie

    @jax.jit
    def fm_targets(key, r0):
      """x1 = GT (from LiePose's r0), x0 ~ the same uniform prior LiePose samples from, s ~ U(0,1)."""
      k1, k2 = jax.random.split(key)
      B = r0.shape[0]
      g1 = jax.vmap(lm.as_lie)(r0)
      g0 = jax.vmap(Lie.sample_uniform)(jax.random.split(k1, B))
      R1 = jax.vmap(lambda g: g.rotation())(g1)
      R0 = jax.vmap(lambda g: g.rotation())(g0)
      t1 = jax.vmap(lambda g: g.translation())(g1)
      t0 = jax.vmap(lambda g: g.translation())(g0)
      vR = jax.vmap(lambda a, b: (a.inverse() @ b).log())(R0, R1)  # body-frame geodesic velocity
      vt = t1 - t0
      sv = jax.random.uniform(k2, (B, 1))
      Rs = jax.vmap(lambda a, v, c: (a @ jaxlie.SO3.exp(c * v)).as_matrix())(R0, vR, sv[:, 0])
      ts = t0 + sv * vt
      return pose_feat_jnp(Rs, ts), sv, jnp.concatenate([vR, vt], -1)

    def get_flat_batch_train(seed, batch, n_slices=256):
      b = base(seed, batch, n_slices=n_slices)
      b["g9"], b["s"], b["v"] = fm_targets(seed, b["r0"])
      return b

    return get_flat_batch_train

  def _fm_update_fn(self):
    model = self.model

    def update_fn(state, batch):
      def loss_fn(params):
        z_ref, z_obs, w = self._inputs(batch["g9"], batch["shape"], batch["tok"].astype(jnp.float32), batch["fg"])
        v = model.apply({"params": params}, batch["g9"], batch["g9"], batch["s"] * 100.0, z_ref, z_obs, w,
                        method=WMDenoiser.noise)
        return jnp.mean(jnp.sum((v - batch["v"]) ** 2, -1))

      loss, grads = jax.value_and_grad(loss_fn)(state.params)
      new_state = state.apply_gradients(grads=grads)
      new_state = new_state.replace(params=freeze({**unfreeze(new_state.params), "wm": state.params["wm"]}))
      return loss, new_state

    return update_fn

  def _fm_sampler(self, n_steps):
    model, lm, Lie = self.model, self.lie_metrics, self.Lie
    params = {"params": self.state.params_ema if self.a.ema_enabled else self.state.params}
    ds = 1.0 / n_steps

    @jax.jit
    def sample(key, img, shape, tok, fg, rt):
      B = rt.shape[0]
      z_ref, z_obs, w = self._inputs(rt, shape, tok, fg)
      g = jax.vmap(lm.as_lie)(rt)
      R = jax.vmap(lambda x: x.rotation())(g)
      t = jax.vmap(lambda x: x.translation())(g)

      def step(carry, k):
        R, t = carry
        g9 = pose_feat_jnp(jax.vmap(lambda x: x.as_matrix())(R), t)
        v = model.apply(params, g9, g9, jnp.full((B, 1), k * ds * 100.0), z_ref, z_obs, w, method=WMDenoiser.noise)
        R = jax.vmap(lambda a, om: a @ jaxlie.SO3.exp(ds * om))(R, v[:, :3])
        return (R, t + ds * v[:, 3:]), None

      (R, t), _ = jax.lax.scan(step, (R, t), jnp.arange(n_steps, dtype=jnp.float32))
      g = jax.vmap(lambda r, tt: Lie.from_rotation_and_translation(r, tt))(R, t)
      return jax.vmap(lambda x: lm.as_repr(x, self.a.repr_type))(g)

    return sample

  # --- train ---

  def create_update_fn(self):
    if self.den_mode in FM_MODES:
      return self._fm_update_fn()
    lie_metrics, loss_name, mu_fn = self.lie_metrics, self.a.loss_name, self._mu_fn()
    assert self.a.learn_noise

    def update_fn(state, batch):
      def loss_fn(params):
        t = batch["t"].reshape((-1, 1)).astype(jnp.float32)
        z_ref, z_obs, w = self._inputs(batch["rt"], batch["shape"], batch["tok"].astype(jnp.float32), batch["fg"])
        mu = mu_fn({"params": params}, batch["rt"], t, z_ref, z_obs, w)
        return jnp.mean(jax.vmap(lie_metrics.get_distance_fn(loss_name))(batch["zt"], mu))

      loss, grads = jax.value_and_grad(loss_fn)(state.params)
      new_state = state.apply_gradients(grads=grads)
      if self.den_mode != "energy":  # world model frozen (restore it: AdamW weight decay would still move it)
        new_state = new_state.replace(params=freeze({**unfreeze(new_state.params), "wm": state.params["wm"]}))
      return loss, new_state

    return update_fn

  # --- sampling ---

  def create_energy_fn(self, t=0):
    """energy mode only: the learned scalar E(g, x, t) at noise level t for final samples (lower = better)."""
    if self.den_mode != "energy":
      return None
    model, lm = self.model, self.lie_metrics
    params = {"params": self.state.params_ema if self.a.ema_enabled else self.state.params}

    @jax.jit
    def energy(rt, shape, tok, fg):
      z_ref, z_obs, w = self._inputs(rt, shape, tok, fg)
      g = jax.vmap(lm.as_lie)(rt)
      g9 = pose_feat_jnp(jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g))
      return model.apply(params, g9, jnp.full((rt.shape[0], 1), t, jnp.float32), z_ref, z_obs, w, method=WMDenoiser.energy)

    return energy

  def create_sampler(self, n_steps=100):
    if self.den_mode in FM_MODES:
      return self._fm_sampler(self.a.get("fm_steps", 20))
    state, mu_fn = self.state, self._mu_fn()
    params = {"params": state.params_ema if self.a.ema_enabled else state.params}
    p_fn = jax.vmap(self.create_p_sample_fn())
    T = self.noise_schedule.timesteps
    cur = (np.linspace(T, 0, n_steps, endpoint=False) - 1).astype(np.int32).tolist()
    seq = jnp.asarray(list(zip(cur, cur[1:] + [0])), jnp.int32)

    @jax.jit
    def sample(key, img, shape, tok, fg, rt):
      B = rt.shape[0]
      z_ref, z_obs, w = self._inputs(rt, shape, tok, fg)

      def step(carry, ttp):
        rt, key = carry
        key, k = jax.random.split(key)
        mu = mu_fn(params, rt, jnp.full((B, 1), ttp[0], jnp.float32), z_ref, z_obs, w)
        _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), ttp[0]), jnp.full((B,), ttp[1]))
        return (rt, key), None

      (rt, _), _ = jax.lax.scan(step, (rt, key), seq)
      return rt

    return sample
