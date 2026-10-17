"""SYMSOL-T SE(3) LiePose testbed with world-model conditioning (see model.py for cond modes).

The world model is the frozen probe-v3 predictor P(z_ref, g) on cached, 2x-pooled DINOv2 tokens.
For `compare`, every noisy pose g_t is turned into ẑ(g_t) = P(z_ref, g_t) and compared with the
observed tokens; the comparison (per-token log error, log energy, pooled residual) conditions the
denoiser for that slice.
"""

import json
import os

import flax
import jax
import jax.numpy as jnp
import numpy as np

from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, FrozenPoseModel, pool_tokens
from liepose.exp.model import broadcast_batch
from liepose.exp.symsolt import run as symsolt_run

from .model import create_cond_model_fn

COND_DIMS = {"none": 1, "pooled": 384, "compare": 64 + 1 + 384, "compare_e": 64 + 1}


class TokenCache:
  """Pre-pooled DINOv2 tokens of one split, indexed like the SymsoltDataset (annotation order)."""

  def __init__(self, cache_dir, split):
    self.tok = np.load(os.path.join(cache_dir, f"{split}_p2_tokens.npy"))
    self.fg = np.load(os.path.join(cache_dir, f"{split}_p2_fg.npy")).astype(np.float32)
    self.label = np.load(os.path.join(cache_dir, f"{split}_p2_meta.npz"))["label"]

  def gather(self, index, label):
    index = np.asarray(index)
    assert (self.label[index] == np.asarray(label)).all(), "token cache misaligned with dataset index"
    return self.tok[index], self.fg[index]


def pose_feat_jnp(R, t):
  return jnp.concatenate([R[:, :, 0], R[:, :, 1], t], axis=-1)


class Testbed(symsolt_run.TestbedSE3):
  def __init__(self, a, exp_path, setup=True):
    self.cond_mode = a.cond_mode
    assert self.cond_mode in COND_DIMS, self.cond_mode
    self.caches = {}
    super().__init__(a, exp_path, setup=setup)
    if setup:
      self.cond_fn = self.create_cond_fn()
      if self.a.get("init_from") and int(self.state.step) == 0:
        self.init_from_checkpoint(self.a.init_from, self.a.get("init_step"))

  def init_from_checkpoint(self, ckpt_dir, step=None):
    """Copy every parameter that exists in a (vanilla LiePose) checkpoint; new layers keep their init."""
    from flax.core import freeze, unfreeze
    from flax.training import checkpoints

    raw = checkpoints.restore_checkpoint(ckpt_dir, target=None, step=step)
    copied, kept = [], []

    def merge(dst, src, path=""):
      out = {}
      for k, v in dst.items():
        if k in src and isinstance(v, dict):
          out[k] = merge(v, src[k], f"{path}/{k}")
        elif k in src:
          assert tuple(np.shape(src[k])) == tuple(np.shape(v)), (path, k, np.shape(src[k]), np.shape(v))
          out[k] = jnp.asarray(src[k])
          copied.append(f"{path}/{k}")
        else:
          out[k] = v
          kept.append(f"{path}/{k}")
      return out

    st = self.state
    self.state = st.replace(params=freeze(merge(unfreeze(st.params), raw["params"])),
                            params_ema=freeze(merge(unfreeze(st.params_ema), raw["params_ema"])),
                            batch_stats=freeze(merge(unfreeze(st.batch_stats), raw["batch_stats"])))
    print(f"init_from {ckpt_dir} step={step}: copied {len(copied)} arrays, kept new init for {sorted(set(kept))}", flush=True)

  # --- model ---

  def create_model(self):
    size = self.a.image_res
    repr_size = self.lie_metrics.get_repr_size(self.a.repr_type)
    fn = create_cond_model_fn(out_dim=repr_size, in_dim=repr_size, cond_dim=COND_DIMS[self.cond_mode],
                              image_shape=(1, size, size, 3), resnet_depth=self.a.resnet_depth,
                              mlp_layers=self.a.mlp_layers, fourier_block=self.a.fourier_block,
                              activ_fn=self.a.activ_fn, use_cond=self.cond_mode != "none",
                              named_cond=bool(self.a.get("named_cond", False)))
    return fn(jax.random.PRNGKey(self.a.seed))

  def cache(self, split):
    if split not in self.caches:
      name = self.a.cache_train if split == "train" else self.a.cache_test
      self.caches[split] = TokenCache(self.a.cache_dir, name)
    return self.caches[split]

  def create_cond_fn(self):
    mode = self.cond_mode
    lm = self.lie_metrics
    if mode == "none":
      return lambda rt, shape, tok, fg: jnp.zeros((rt.shape[0], 1), jnp.float32)

    if mode == "pooled":
      def cond(rt, shape, tok, fg):
        w = FG_FLOOR + fg
        pooled = (w[..., None] * tok).sum(1) / w.sum(1, keepdims=True)
        return broadcast_batch(pooled, rt.shape[0])
      return cond

    with open(os.path.join(self.a.wm_dir, "args.json")) as f:
      pa = json.load(f)
    assert pa["pool"] == 2, "token cache is 2x-pooled"
    refs, _ = pool_tokens(np.load(os.path.join(self.a.cache_dir, "refs_tokens.npy")),
                          np.load(os.path.join(self.a.cache_dir, "refs_fg.npy")), 2)
    refs = jnp.asarray(refs, jnp.float32)
    wm = FrozenPoseModel(dim=pa["dim"], depth=pa["depth"], heads=pa["dim"] // 64, out_dim=refs.shape[-1])
    init = wm.init(jax.random.PRNGKey(0), jnp.zeros((1,) + refs.shape[1:]), jnp.zeros((1, 9)))
    with open(os.path.join(self.a.wm_dir, "params.msgpack"), "rb") as f:
      wm_params = flax.serialization.from_bytes(init, f.read())

    def cond(rt, shape, tok, fg):
      bs, b = rt.shape[0], tok.shape[0]
      s = bs // b
      g = jax.vmap(lm.as_lie)(rt)
      R = jax.vmap(lambda x: x.rotation().as_matrix())(g)
      t = jax.vmap(lambda x: x.translation())(g)
      z_ref = jnp.repeat(refs[shape], s, axis=0)
      pred = wm.apply(wm_params, z_ref, pose_feat_jnp(R, t))
      tgt = jnp.repeat(tok, s, axis=0)
      w = jnp.repeat(FG_FLOOR + fg, s, axis=0)
      r = pred - tgt
      per_tok = (r ** 2).mean(-1)
      energy = (w * per_tok).sum(-1) / w.sum(-1)
      feats = [jnp.log(per_tok + 1e-4), jnp.log(energy + 1e-6)[:, None]]
      if mode == "compare":
        feats.append((w[..., None] * r).sum(1) / w.sum(-1, keepdims=True))
      return jax.lax.stop_gradient(jnp.concatenate(feats, axis=-1))

    return cond

  # --- train ---

  def create_get_flat_batch_train_fn(self):
    base = super().create_get_flat_batch_train_fn()
    cache = self.cache("train")

    def get_flat_batch_train(seed, batch, n_slices=256):
      b = base(seed, batch, n_slices=n_slices)
      tok, fg = cache.gather(batch["index"], batch["label_shape"])
      b["tok"] = jnp.asarray(tok)
      b["fg"] = jnp.asarray(fg)
      b["shape"] = jnp.asarray(batch["label_shape"], jnp.int32)
      return b

    return get_flat_batch_train

  def create_update_fn(self):
    lie_metrics = self.lie_metrics
    learn_noise = self.a.learn_noise
    loss_name = self.a.loss_name
    cond_fn = self.cond_fn
    gate_t = self.a.get("cond_gate_t", -1)

    def update_fn(state, batch):
      def loss_fn(params, batch):
        t = batch["t"].reshape((-1, 1))
        c = cond_fn(batch["rt"], batch["shape"], batch["tok"].astype(jnp.float32), batch["fg"])
        c = c * (t >= gate_t)  # world-model signal only at high/mid noise (gate_t < 0: always on)
        mu, new_state = state.apply_fn({"params": params, "batch_stats": state.batch_stats},
                                       batch["img"], batch["rt"], t, c, mutable=["batch_stats"])
        ta = batch["zt"] if learn_noise else batch["r0"]
        loss = jax.vmap(lie_metrics.get_distance_fn(loss_name))(ta, mu)
        return jnp.mean(loss), new_state

      (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
      state = state.apply_gradients(grads=grads, batch_stats=new_state["batch_stats"])
      return loss, state

    return update_fn

  def run_train_single(self):
    # resume: the LR schedule lives in the restored optimizer state; only the loop length changes
    total, done = self.a.train_steps, int(self.state.step)
    if done >= total:
      print(f"already trained {done} steps")
      return self
    self.a.train_steps = total - done
    try:
      return super().run_train_single()
    finally:
      self.a.train_steps = total

  # --- sampling ---

  def create_sampler(self, n_steps=100):
    """Returns sample(key, img, shape, tok, fg, rt0) -> rt (final poses, repr)."""
    state, model = self.state, self.model
    ema = self.a.ema_enabled
    params = state.params_ema if ema else state.params
    cond_fn = self.cond_fn
    gate_t = self.a.get("cond_gate_t", -1)
    p_fn = jax.vmap(self.create_p_sample_fn())
    T = self.noise_schedule.timesteps
    cur = (np.linspace(T, 0, n_steps, endpoint=False) - 1).astype(np.int32).tolist()
    seq = jnp.asarray(list(zip(cur, cur[1:] + [0])), jnp.int32)

    @jax.jit
    def sample(key, img, shape, tok, fg, rt):
      feat = model.backbone.apply({"params": params["backbone"], "batch_stats": state.batch_stats["backbone"]},
                                  img, mutable=False)
      B = rt.shape[0]

      def step(carry, ttp):
        rt, key = carry
        t, tp = ttp[0], ttp[1]
        key, k = jax.random.split(key)
        c = cond_fn(rt, shape, tok, fg) * (t >= gate_t)
        mu = model.head.apply({"params": params["head"]}, feat, rt, jnp.full((B, 1), t, jnp.float32), c)
        _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
        return (rt, key), None

      (rt, _), _ = jax.lax.scan(step, (rt, key), seq)
      return rt

    return sample
