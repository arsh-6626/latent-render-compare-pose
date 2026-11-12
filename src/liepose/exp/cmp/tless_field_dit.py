"""T-LESS SE(3) diffusion with the feature-field world model (tless_field.TlessField) in the loop: sp_field_dit's
denoiser on tless_dit's T-LESS setup (LiePose's crops, BOP19 test, Evaluator).

At every denoising step the frozen field renders the current hypothesis of the target object (obj_id from the BOP
task) in the observation's own crop camera; the denoiser compares it with the observed tokens (sp_field_dit modes).
The crop meta (box + intrinsics, tless_field.crop_meta10) enters the render only: the denoiser has no crop
conditioning (use_k=False), as tless_dit and LiePose, so field vs retrieval world model differ in the world model alone.

  token-cache modes  crop meta per cache row from {cache}/{split}_crop_meta.npy (tless_field.tless_crop_meta of
                     export_crops.py --meta shards aligned with the cache)
  *_ft               LiePose's live image pipeline (tless_dit dit_ft): meta from the batch, image x visible mask

  DATADIR=... BOP_RENDERER=vispy python -m liepose.exp.cmp.tless_field_dit --root_path $LOGS/experiments/tless-field \
      "den_mode=[field_full_lat]" "field_dir=[$LOGS/probe/tl_field_v1]" "train_steps=[100000]"
"""

import json
import os
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze
from jaxlie import R3SO3, SE3, SO3
from tqdm import tqdm

from liepose.exp.bop import run as bop_run
from liepose.exp.bop.ds_config import DS
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.probe_tless import TokenSplit
from liepose.exp.jepa_probe.tless_field import crop_meta10, load_tless_field
from liepose.exp.jepa_probe.wm_eval import app_stats
from liepose.utils.config import get_args, load_config_from_args, save_config
from liepose.utils.exploration import Options

from . import tless_dit
from .sp_field_dit import BASE_MODES, FieldDenoiser, FtFieldDenoiser
from .tless_dit import mask_fg

MODES = BASE_MODES + tuple(m + "_ft" for m in BASE_MODES)
SAMPLE = tless_dit.SAMPLE


class Testbed(tless_dit.Testbed):

  def __init__(self, a, exp_path, setup=True):
    assert a.den_mode in MODES, a.den_mode
    self.ft = a.den_mode.endswith("_ft")
    self.base_mode = a.den_mode[:-3] if self.ft else a.den_mode
    bop_run.TestbedSE3.__init__(self, a, exp_path, setup=setup)

  def create_model(self):
    D = os.environ["DATADIR"]
    self.cache = self.a.tl_cache or os.path.join(D, DS["tokens"])
    self.train_split = TokenSplit(self.cache, "train")
    if not self.ft:
      self.train_meta = np.load(os.path.join(self.cache, "train_crop_meta.npy")).astype(np.float32)
      assert len(self.train_meta) == len(self.train_split.obj)
    self.field_model, self.field_params = load_tless_field(self.a.field_dir)
    n, z = 2, jnp.zeros((2, 64, 384))
    hz = jnp.zeros((n, 64, self.field_model.hid))
    k2 = jnp.zeros((n, 10))
    if self.ft:
      model = FtFieldDenoiser(mode=self.base_mode, dim=self.a.dit_dim, depth=self.a.dit_depth, use_k=False)
      params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 224, 224, 3)), jnp.zeros((n, 6)), jnp.zeros((n, 1)), k2,
                                   jnp.ones((n, 64)), jnp.ones((n, 64)), hz, method=FtFieldDenoiser.full_init))
      dino = flax.serialization.msgpack_restore(open(os.path.join(D, "dinov2_vits14_224.msgpack"), "rb").read())["params"]
      shapes = lambda p: jax.tree_util.tree_map(np.shape, p)
      for name in ("dino", "dino_frozen"):
        assert shapes(dino) == shapes(params["params"][name]), "DINOv2 checkpoint does not match"
        params["params"][name] = jax.tree_util.tree_map(jnp.asarray, dino)
    else:
      model = FieldDenoiser(mode=self.base_mode, dim=self.a.dit_dim, depth=self.a.dit_depth, use_k=False)
      params = unfreeze(model.init(jax.random.PRNGKey(self.a.seed), jnp.zeros((n, 6)), jnp.zeros((n, 1)), k2,
                                   z, jnp.ones((n, 64)), z, jnp.ones((n, 64)), hz))
    if self.a.joint:  # A1: the comparison space is trained, so the field is trained with it (field loss, update_fn)
      assert self.ft, "joint: *_ft modes only"
      params["params"]["field"] = unfreeze(self.field_params)["params"]
    params["batch_stats"] = {}
    return model, FrozenDict(params)

  def _render(self, obj, Rg, tg, kg, z_obs, fparams=None):
    """field of each hypothesis' object at (allocentric R, translation_scaled) in the observation's crop; never trained
    through the denoiser (stop-gradient); joint: the field's current (or EMA) parameters"""
    return jax.lax.stop_gradient(self.field_model.apply(fparams or self.field_params, obj, Rg, tg, kg, app_stats(z_obs), with_latent=True))

  @staticmethod
  def _split_field(params):
    """joint: (denoiser params without the field, {"params": field params} or None)"""
    if "field" not in params:
      return params, None
    return {k: v for k, v in params.items() if k != "field"}, {"params": params["field"]}

  # --- train ---

  def create_sample_train_fn(self):
    """process=fm: flow matching on SE(3) instead of LiePose's diffusion. The pose moves on the geodesic from the data
    pose g0 (tau=0) to a prior pose g1 (tau=1, the sampler's prior: uniform rotation, translation in [-1, 1]^3),
    g_tau = g0 Exp(tau xi), xi = Log(g0^-1 g1); the network predicts the constant body-frame velocity xi. t = tau * T
    keeps the step embedding on the diffusion's scale."""
    if self.a.process == "diffusion":
      return super().create_sample_train_fn()
    assert self.a.process in FM, self.a.process
    Lie, lm, T, rep = self.Lie, self.lie_metrics, self.noise_schedule.timesteps, self.a.repr_type
    if self.a.process == "fmr3":  # SO(3) x R^3: rotation geodesic, straight translation line (velocity [t1 - t0, omega])
      def sample_train_fn(seed, rot, tran):
        k1, k2 = jax.random.split(seed)
        tau = jax.random.uniform(k2, ())
        g0 = R3SO3.from_rotation_and_translation(SO3(rot), tran)
        xi = (g0.inverse() @ R3SO3.sample_uniform(k1)).log()
        gt = g0 @ R3SO3.exp(tau * xi)
        return {"rt": lm.as_repr(_se3(gt), rep), "t": tau * T, "zt": xi, "r0": lm.as_repr(_se3(g0), rep)}

      return sample_train_fn

    def sample_train_fn(seed, rot, tran):
      k1, k2 = jax.random.split(seed)
      tau = jax.random.uniform(k2, ())
      g0 = Lie.from_rotation_and_translation(SO3(rot), tran)
      xi = (g0.inverse() @ Lie.sample_uniform(k1)).log()
      gt = g0 @ Lie.exp(tau * xi)
      return {"rt": lm.as_repr(gt, rep), "t": tau * T, "zt": xi, "r0": lm.as_repr(g0, rep)}

    return sample_train_fn

  def create_get_flat_batch_train_fn(self):
    base = bop_run.TestbedSE3.create_get_flat_batch_train_fn(self)  # LiePose noising of (rotation_allo, translation_scaled)
    poses = self._poses_fn()
    d = self.train_split

    def get_flat_batch_train(seed, batch, n_slices=256):
      if self.ft:  # images go to the backbones once per image (masked as the cache), not once per slice
        batch = dict(batch)
        mask = np.asarray(batch["mask"], np.float32)
        # fg_uniform: no visible mask reaches the denoiser, so the images are not masked either (as export_crops --no_mask)
        image = np.asarray(batch["image"], np.float32) * (1.0 if self.a.fg_uniform else mask)
        meta = crop_meta10({k: np.asarray(batch[k]) for k in ("bbox", "bbox_center", "dzi_size", "focal_ratio", "cam")})
        obj = np.asarray(batch["object_id"])
        batch["image"], batch["mask"] = np.zeros((len(image), 1, 1, 3), np.float32), np.ones((len(image), 1, 1, 1), np.float32)
      else:
        row = batch["row"]
        obj, meta = d.obj[row], self.train_meta[row]
      b = base(seed, batch, n_slices=n_slices)
      b.pop("img", None)
      s = b["rt"].shape[0] // len(obj)
      Rg, tg = poses(b["rt"])
      b.update(Rg=Rg, tg=tg, kg=jnp.asarray(np.repeat(meta, s, 0)), obj=jnp.asarray(np.repeat(obj, s)))
      if self.ft:
        b.update(image=jnp.asarray(image), fg=mask_fg(jnp.asarray(mask[..., 0])))
        if self.a.joint:  # GT pose of each image (allocentric R, translation_scaled), for the field's fitting loss
          q = np.asarray(batch["rotation_allo"], np.float32)
          b.update(R0=jax.vmap(lambda x: SO3(x).as_matrix())(jnp.asarray(q)), t0=jnp.asarray(np.asarray(batch["translation_scaled"], np.float32)),
                   k0=jnp.asarray(meta), obj0=jnp.asarray(obj), fg0=b["fg"])  # fg0: the training visible mask (kept under fg_uniform)
      else:
        b.update(tok=jnp.asarray(d.tokens[row]), fg=jnp.asarray(d.fg[row]))
      if self.a.fg_uniform:  # no visible mask reaches the denoiser (the cache's train fg only weights the field's loss)
        b["fg"] = jnp.ones_like(b["fg"])
      return b

    return get_flat_batch_train

  def create_update_fn(self):
    lie_metrics, loss_name, model = self.lie_metrics, self.a.loss_name, self.model
    assert self.a.learn_noise
    fm = self.a.process in FM
    huber = self.a.process == "fmr3"

    def loss_fn(params, mb):
      """mb: (part of) a batch; per-image keys (fg, tok / image, keep) and per-slice keys, slices image-major"""
      t = mb["t"].reshape((-1, 1)).astype(jnp.float32)
      s = mb["rt"].shape[0] // mb["fg"].shape[0]
      w = jnp.repeat(FG_FLOOR + mb["fg"], s, 0)
      keep = jnp.repeat(mb["keep"], s, 0) if "keep" in mb else None
      z_ext, f_loss = None, 0.0
      params, fparams = self._split_field(params)
      if "image" in mb:  # frozen (joint: EMA) copy -> z_obs (the field's comparison space), trainable copy -> z_ext
        z_img, z_ext = model.apply({"params": params}, mb["image"], method=FtFieldDenoiser.encode)
        z_ext = jnp.repeat(z_ext, s, 0)
      else:
        z_img = mb["tok"].astype(jnp.float32)
      if fparams is not None:  # the field's fitting loss at the GT pose, towards the (stop-gradient) EMA tokens
        pred = self.field_model.apply(fparams, mb["obj0"], mb["R0"], mb["t0"], mb["k0"], app_stats(z_img))
        f_loss = weighted_energy(pred, z_img, FG_FLOOR + mb.get("fg0", mb["fg"])).mean()
      z_obs = jnp.repeat(z_img, s, 0)
      z_hat, opa, h = self._render(mb["obj"], mb["Rg"], mb["tg"], mb["kg"], z_obs, fparams)
      mu = model.apply({"params": params}, mb["rt"], t, mb["kg"], z_obs, w, z_hat, opa, h, keep, z_ext)
      if huber:  # [translation | rotation] velocity: pseudo-Huber on translation (train tails reach |t| ~ 47), MSE on rotation
        r = mu - mb["zt"]
        d = self.a.fm_delta
        return jnp.mean(jnp.sum(d ** 2 * (jnp.sqrt(1 + (r[:, :3] / d) ** 2) - 1), -1) + jnp.sum(r[:, 3:] ** 2, -1))
      if fm:  # velocity regression in se(3)
        return jnp.mean(jnp.sum((mu - mb["zt"]) ** 2, -1))
      return jnp.mean(jax.vmap(lie_metrics.get_distance_fn(loss_name))(mb["zt"], mu)) + self.a.joint_w * f_loss

    def update_fn(state, batch):
      keys = ("rt", "t", "zt", "Rg", "tg", "kg", "obj", "fg") + tuple(k for k in ("tok", "image", "R0", "t0", "k0", "obj0", "fg0") if k in batch)
      mb = {k: batch[k] for k in keys}
      n_img = mb["fg"].shape[0]
      if self.base_mode == "field_drop":  # per image (all its slices), from the step's noise draws (no extra rng state)
        u = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(0), state.step), (n_img,))
        mb["keep"] = (u >= self.a.drop_p).astype(jnp.float32)
      m = self.a.grad_accum
      if m == 1:
        loss, grads = jax.value_and_grad(loss_fn)(state.params, mb)
      else:  # the same mean loss over m image-major chunks, sequentially (sp_field_dit.update_fn)
        assert n_img % m == 0, (n_img, m)
        chunks = jax.tree_util.tree_map(lambda x: x.reshape((m, x.shape[0] // m) + x.shape[1:]), mb)

        def acc(carry, c):
          l, g = jax.value_and_grad(loss_fn)(state.params, c)
          return jax.tree_util.tree_map(jnp.add, carry, (l, g)), None

        zero = (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, state.params))
        (loss, grads), _ = jax.lax.scan(acc, zero, chunks)
        loss, grads = jax.tree_util.tree_map(lambda x: x / m, (loss, grads))
      state = state.apply_gradients(grads=grads)
      if self.a.joint:  # the comparison space follows the trained encoder slowly (EMA target, BYOL / JEPA style)
        p, tau = unfreeze(state.params), self.a.joint_tau
        p["dino_frozen"] = jax.tree_util.tree_map(lambda f, d: tau * f + (1 - tau) * d, p["dino_frozen"], p["dino"])
        state = state.replace(params=freeze(p) if isinstance(state.params, FrozenDict) else p)
      return loss, state

    return update_fn

  # --- test ---

  def create_sampler(self, n_steps=100):
    model, poses = self.model, self._poses_fn()
    params = {"params": self.state.params_ema if self.a.ema_enabled else self.state.params}
    params["params"], fp = self._split_field(params["params"])
    if fp is not None:
      self.field_params = fp  # joint: the trained field (run_select / energies use self._render)
    T = self.noise_schedule.timesteps
    fm = self.a.process in FM
    if fm:  # Euler steps of the ODE on the group, tau 1 -> 0: g <- g Exp(-(tau - tau') v)
      lm, rep = self.lie_metrics, self.a.repr_type
      taus = np.linspace(1.0, 0.0, n_steps + 1)
      seq = [(float(a) * T, float(a - b)) for a, b in zip(taus[:-1], taus[1:])]
      if self.a.process == "fmr3":
        move = jax.vmap(lambda r, v, d: lm.as_repr(_se3(_r3so3(lm.as_lie(r)) @ R3SO3.exp(-d * v)), rep), (0, 0, None))
      else:
        move = jax.vmap(lambda r, v, d: lm.as_repr(lm.as_lie(r) @ self.Lie.exp(-d * v), rep), (0, 0, None))
    else:
      p_fn = jax.vmap(self.create_p_sample_fn())
      cur = (np.linspace(T, 0, n_steps, endpoint=False) - 1).astype(np.int32).tolist()  # = Testbed.run_test
      seq = list(zip(cur, cur[1:] + [0]))

    @jax.jit
    def step(key, rt, t, tp, obj, z_obs, w, kg, z_ext):
      """diffusion: t, tp = step and previous step; fm: t = tau * T, tp = the step size in tau"""
      B = rt.shape[0]
      key, k = jax.random.split(key)
      Rg, tg = poses(rt)
      z_hat, opa, h = self._render(obj, Rg, tg, kg, z_obs)
      mu = model.apply(params, rt, jnp.full((B, 1), t, jnp.float32), kg, z_obs, w, z_hat, opa, h, None, z_ext)
      if fm:
        return key, move(rt, mu, tp)
      _, rt = p_fn(jax.random.split(k, B), mu, rt, jnp.full((B,), t), jnp.full((B,), tp))
      return key, rt

    encode = jax.jit(lambda image: model.apply(params, image, method=FtFieldDenoiser.encode)) if self.ft else None

    def sample(key, rt, obj, tok, fg, meta, image=None, z_ext=None):
      """tok: the cached tokens, or (ft) None with the masked crop images, encoded once per image; or (ft, run_select)
      tok and z_ext already encoded (self._encode) and repeated per chain"""
      if image is not None:
        tok, z_ext = encode(image)
      obj = jnp.asarray(obj)
      for t, tp in seq:
        t, tp = (jnp.float32(t), jnp.float32(tp)) if fm else (jnp.int32(t), jnp.int32(tp))
        key, rt = step(key, rt, t, tp, obj, tok, fg, meta, z_ext)
      return rt

    self._encode = encode
    return sample

  def run_test(self, opt, opt_path, limit=None):
    """tless_dit.run_test with the crop meta (the test cache's rows, or the batch's for *_ft)"""
    self.opt, self.opt_path = opt, opt_path
    os.makedirs(opt_path, exist_ok=True)
    obs = TokenSplit(self.cache, "test")
    test_meta = None if self.ft else np.load(os.path.join(self.cache, "test_crop_meta.npy")).astype(np.float32)
    row_of = {int(i): r for r, i in enumerate(obs.index)}
    assert len(row_of) == len(obs.index)
    dataset = self.create_test_dataset()
    assert len(dataset.dataset) == len(obs.index), (len(dataset.dataset), len(obs.index))
    evaluator = self.create_test_evaluator(dataset)
    lm, Lie = self.lie_metrics, self.Lie
    prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), self.a.repr_type)))
    sample = self.create_sampler(opt.steps)
    key, n = jax.random.PRNGKey(self.a.seed + 2), 0
    for batch in tqdm(dataset, desc="Sample"):
      rows = np.array([row_of[int(i)] for i in np.asarray(batch["index"])])
      assert (obs.obj[rows] == np.asarray(batch["object_id"])).all(), "token cache misaligned with the test dataset"
      key, k1, k2 = jax.random.split(key, 3)
      t0 = time.time()
      if self.ft:  # fg_uniform (or detection crops, which carry no mask): unmasked images, uniform weights
        img = np.asarray(batch["image"], np.float32)
        mask = np.ones(img.shape[:3] + (1,), np.float32) if self.a.fg_uniform or "mask" not in batch else \
          np.asarray(batch["mask"], np.float32)
        image = jnp.asarray(img * mask)
        meta = crop_meta10({k: np.asarray(batch[k]) for k in ("bbox", "bbox_center", "dzi_size", "focal_ratio", "cam")})
        rt = sample(k2, prior(jax.random.split(k1, len(rows))), obs.obj[rows], None, FG_FLOOR + mask_fg(jnp.asarray(mask[..., 0])),
                    jnp.asarray(meta), image)
      else:
        fg = np.ones_like(obs.fg[rows]) if self.a.fg_uniform else obs.fg[rows]
        rt = sample(k2, prior(jax.random.split(k1, len(rows))), obs.obj[rows], jnp.asarray(obs.tokens[rows], jnp.float32),
                    jnp.asarray(FG_FLOOR + fg), jnp.asarray(test_meta[rows]))
      rt = jax.block_until_ready(rt)
      dt = time.time() - t0
      seq_rt = self.post_process(rt)[None, :, None]
      evaluator.process(batch, {"seq_r0": seq_rt, "seq_rt": seq_rt, "time": dt})
      n += len(rows)
      if limit is not None and n >= limit:
        break
    evaluator.summarize()
    return n

  def run_select(self, opt, opt_path, S=16, Ns=(1, 4, 16), limit=None, depth_grid=None):
    """Selection among N samples (paper tab:select / "+ selection" rows), token-cache modes. Each crop gets S independent
    chains of the plain sampler (same sampler as run_test); the first N of them are the candidates for N in Ns. Rules:
      consensus  the sample with the most others within 10 deg (symmetry-aware geodesic; ties: smaller mean distance),
                 as speedplus/consensus_score.py; no ground truth, no world model
      energy     the lowest field energy at the final pose (eq:energy: foreground-weighted squared token distance between
                 the field's render and the observed tokens); no ground truth
      oracle     the smallest symmetry-aware rotation error to the ground truth (diagnostic only)
    N = 1 is the first sample for every rule. Writes one BOP19 csv per (rule, N) under opt_path/{rule}_n{N} (N = 1:
    opt_path/n1) and all samples with their energies to opt_path/samples.npz; inference reads the crop's tokens, mask
    and box only, ground truth is read by the oracle rule alone.
    depth_grid (lo, hi, K): also n1_dz, sample 0 with its depth replaced by the lowest-energy one among K factors in
    [lo, hi] of its metric depth (rotation and image-plane position kept; the scaled depth of LiePose's translation is
    linear in the metric depth: z' = s (z + off) - off). GT-free."""
    from scipy.spatial.transform import Rotation

    from liepose.exp.jepa_probe.probe_tless import load_symmetries
    from liepose.data.bop.evaluator import wxyz_to_xyzw
    self.opt = opt
    obs = TokenSplit(self.cache, "test")
    test_meta = None if self.ft else np.load(os.path.join(self.cache, "test_crop_meta.npy")).astype(np.float32)
    row_of = {int(i): r for r, i in enumerate(obs.index)}
    dataset = self.create_test_dataset()
    assert len(dataset.dataset) == len(obs.index), (len(dataset.dataset), len(obs.index))
    Ns = tuple(N for N in Ns if N <= S)
    names = ["n1"] + [f"{r}_n{N}" for N in Ns if N > 1 for r in ("consensus", "energy", "oracle")]
    if depth_grid:
      names.append("n1_dz")
      dz = np.linspace(*depth_grid[:2], int(depth_grid[2])).astype(np.float32)
      off = jnp.float32(self.a.tran_offset[2])
    evals = {}
    for nm in names:
      self.opt_path = os.path.join(opt_path, nm)
      os.makedirs(self.opt_path, exist_ok=True)
      evals[nm] = self.create_test_evaluator(dataset)
    self.opt_path = opt_path
    D = os.environ["DATADIR"]
    syms = load_symmetries(os.path.join(D, DS["models_info"]))
    lm, Lie, poses = self.lie_metrics, self.Lie, self._poses_fn()
    prior = jax.jit(jax.vmap(lambda k: lm.as_repr(Lie.sample_uniform(k), self.a.repr_type)))
    sample = self.create_sampler(opt.steps)

    @jax.jit
    def energy(rt, obj, z_obs, w, kg):
      Rg, tg = poses(rt)
      z_hat, _, _ = self._render(obj, Rg, tg, kg, z_obs)
      e = ((z_hat - z_obs) ** 2).mean(-1)
      return (w * e).sum(-1) / w.sum(-1)

    @jax.jit
    def energy_t(Rg, tg, obj, z_obs, w, kg):
      z_hat, _, _ = self._render(obj, Rg, tg, kg, z_obs)
      e = ((z_hat - z_obs) ** 2).mean(-1)
      return (w * e).sum(-1) / w.sum(-1)

    def sym_dist(Ra, Rb, sy):
      """(a,3,3), (b,3,3), (m,3,3) -> (a,b) min over symmetries of the geodesic angle (deg) between Ra and Rb @ s"""
      Rbs = np.einsum("bij,mjk->bmik", Rb, sy)
      c = (np.einsum("aij,bmij->abm", Ra, Rbs) - 1) / 2
      return np.degrees(np.arccos(np.clip(c, -1, 1))).min(-1)

    key, n = jax.random.PRNGKey(self.a.seed + 2), 0
    out_idx, out_pose, out_e, out_dz = [], [], [], []
    for batch in tqdm(dataset, desc="Select"):
      rows = np.array([row_of[int(i)] for i in np.asarray(batch["index"])])
      assert (obs.obj[rows] == np.asarray(batch["object_id"])).all(), "token cache misaligned with the test dataset"
      B = len(rows)
      rr = np.repeat(rows, S)  # crop-major: the S chains of a crop are consecutive
      key, k1, k2 = jax.random.split(key, 3)
      t0 = time.time()
      obj = obs.obj[rr]
      if self.ft:  # as run_test: the crop image (x visible mask) through the trained encoder, once per crop
        img = np.asarray(batch["image"], np.float32)
        mask = np.ones(img.shape[:3] + (1,), np.float32) if self.a.fg_uniform or "mask" not in batch else \
          np.asarray(batch["mask"], np.float32)
        tok0, ze0 = self._encode(jnp.asarray(img * mask))
        meta = crop_meta10({k: np.asarray(batch[k]) for k in ("bbox", "bbox_center", "dzi_size", "focal_ratio", "cam")})
        tok, ze = jnp.repeat(tok0, S, 0), jnp.repeat(ze0, S, 0)
        w = jnp.repeat(FG_FLOOR + mask_fg(jnp.asarray(mask[..., 0])), S, 0)
        kg = jnp.repeat(jnp.asarray(meta), S, 0)
      else:
        tok, ze = jnp.asarray(obs.tokens[rr], jnp.float32), None
        w, kg = jnp.asarray(FG_FLOOR + obs.fg[rr]), jnp.asarray(test_meta[rr])
      rt = sample(k2, prior(jax.random.split(k1, B * S)), obj, tok, w, kg, None, ze)
      e = np.asarray(energy(rt, jnp.asarray(obj), tok, w, kg)).reshape(B, S)
      dt = (time.time() - t0) / S  # per-sample time, as one run_test sample
      pose = np.asarray(self.post_process(rt)).reshape(B, S, 7)
      if depth_grid:  # sample 0 of each crop: energy over depth factors, rotation and image-plane position fixed
        r0 = np.arange(B) * S
        Rg0, tg0 = poses(rt[r0])
        o0, z0, w0, k0 = jnp.asarray(obj[r0]), tok[r0], w[r0], kg[r0]
        ed = np.stack([np.asarray(energy_t(Rg0, tg0.at[:, 2].set(f * (tg0[:, 2] + off) - off), o0, z0, w0, k0)) for f in dz], 1)
        best = dz[ed.argmin(1)]
        pose_dz = pose[:, 0].copy()
        pose_dz[:, 6] = best * (pose_dz[:, 6] + float(off)) - float(off)
        out_dz.append(best)
      out_idx.append(np.asarray(batch["index"])); out_pose.append(pose); out_e.append(e)
      for b in range(B):
        inp = {k: (v[b] if hasattr(v, "__len__") and len(v) == B else v) for k, v in batch.items()}
        Rs = np.stack([evals["n1"]._process_pose(inp, pose[b, s])[0] for s in range(S)])
        sy = syms[int(inp["object_id"])]
        G = sym_dist(Rs, Rs, sy)
        gt = Rotation.from_quat(wxyz_to_xyzw(np.asarray(inp["rotation"], np.float64))).as_matrix()
        err = sym_dist(gt[None], Rs, sy)[0]
        pick = {"n1": 0}
        for N in Ns:
          if N == 1:
            continue
          g = G[:N, :N]
          pick[f"consensus_n{N}"] = int(((g < 10).sum(-1) - 1e-3 * g.mean(-1)).argmax())
          pick[f"energy_n{N}"] = int(e[b, :N].argmin())
          pick[f"oracle_n{N}"] = int(err[:N].argmin())
        one = {k: (v[b:b + 1] if hasattr(v, "__len__") and len(v) == B else v) for k, v in batch.items()}
        if depth_grid:
          sel = pose_dz[b][None, None, None]
          evals["n1_dz"].process(one, {"seq_r0": sel, "seq_rt": sel, "time": dt * (1 + len(dz))})
        for nm, j in pick.items():
          sel = pose[b, j][None, None, None]
          evals[nm].process(one, {"seq_r0": sel, "seq_rt": sel, "time": dt * (1 if nm == "n1" else int(nm.rsplit("n", 1)[1]))})
      n += B
      if limit is not None and n >= limit:
        break
    np.savez(os.path.join(opt_path, "samples.npz"), index=np.concatenate(out_idx), pose=np.concatenate(out_pose),
             energy=np.concatenate(out_e), **({"depth_factor": np.concatenate(out_dz)} if depth_grid else {}))
    for nm, ev in evals.items():
      ev.summarize()
    return n, names


FM = ("fm", "fmr3")


def _se3(g):
  return SE3.from_rotation_and_translation(g.rotation(), g.translation())


def _r3so3(g):
  return R3SO3.from_rotation_and_translation(g.rotation(), g.translation())


def build_explorations():
  e = tless_dit.build_explorations()
  e.den_mode = ["field_full_lat"]
  e.field_dir = [None]
  e.drop_p = [0.5]
  e.tl_cache = [None]  # token cache with {split}_crop_meta.npy (default $DATADIR/tless_tokens)
  e.grad_accum = [1]  # micro-batches per step (same gradient, less memory; sp_field_dit)
  e.process = ["diffusion"]  # diffusion (LiePose's) | fm (flow on SE(3)) | fmr3 (flow on SO(3) x R^3); create_sample_train_fn
  e.fm_delta = [1.0]
  e.fg_uniform = [False]  # True: uniform token weights in training and test (no-mask caches, export_crops --no_mask)  # fmr3: pseudo-Huber scale of the translation velocity loss
  e.fm_steps = ["10/20/100"]  # fm: ODE steps of the test runs (one BOP19 test each)
  e.depth_grid = [None]  # "lo/hi/K" with select_n: also n1_dz (field-energy depth search on sample 0)
  e.select_n = [0]
  e.joint = [False]  # A1 (*_ft): train the comparison space (EMA of the trained DINOv2) and the field with the denoiser
  e.joint_tau = [0.999]  # EMA decay of the comparison encoder
  e.joint_w = [1.0]  # weight of the field's fitting loss
  return e


def main():
  a = get_args(exp_name="tless-field", desc="Feature-field world model in T-LESS SE(3) diffusion")
  for opt in load_config_from_args(a, build_explorations()).expand():
    opt.pop("sample_options", None)
    field = os.path.basename(os.path.normpath(opt.field_dir))
    exp_path = os.path.join(a.root_path, opt.to_string(includes=["den_mode", "train_steps", "seed"]) + f"+field={field}"
                            + ("" if opt.process == "diffusion" else f"+process={opt.process}")
                            + (f"+joint={opt.joint_tau}" if opt.joint else ""))
    tb = Testbed(opt, exp_path, setup=True)
    if not opt.skip_train:
      save_config(opt, os.path.join(exp_path, "configs.yaml"))
      tb.run_train()
    step = int(tb.state.step)
    if opt.select_n:
      out = os.path.join(exp_path, f"test_{step}_sel{opt.select_n}" + ("_dz" if opt.depth_grid else ""))
      dg = [float(x) for x in str(opt.depth_grid).split("/")] if opt.depth_grid else None
      sel_sample = dict(SAMPLE, batch_size=int(os.environ.get("SEL_BS", SAMPLE["batch_size"])))  # crops per batch (memory only)
      n, names = tb.run_select(Options(sel_sample), out, S=int(opt.select_n), limit=opt.test_limit, depth_grid=dg)
      for nm in names:
        tb.run_eval(Options(SAMPLE), os.path.join(out, nm))
      json.dump({"exp": exp_path, "step": step, "n_instances": n, "sample": SAMPLE, "select_n": int(opt.select_n), "depth_grid": opt.depth_grid,
                 "runs": names, "renderer": os.environ.get("BOP_RENDERER", "cpp"), "detections": os.environ.get("BOP_DET"),
                 "fg_uniform": bool(opt.fg_uniform)}, open(os.path.join(out, "run.json"), "w"), indent=2)
      continue
    runs = [(f"test_{step}", SAMPLE)] if opt.process == "diffusion" else \
           [(f"test_{step}_n{n}", dict(SAMPLE, steps=int(n))) for n in str(opt.fm_steps).split("/")]
    for name, sample in runs:
      out = os.path.join(exp_path, name)
      n = tb.run_test(Options(sample), out, limit=opt.test_limit)
      tb.run_eval(Options(sample), out)
      json.dump({"exp": exp_path, "step": step, "n_instances": n, "sample": sample, "process": opt.process,
                 "renderer": os.environ.get("BOP_RENDERER", "cpp"), "detections": os.environ.get("BOP_DET"),
                 "train_box": os.environ.get("BOP_TRAIN_BOX", "obj"), "fg_uniform": bool(opt.fg_uniform)}, open(os.path.join(out, "run.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
