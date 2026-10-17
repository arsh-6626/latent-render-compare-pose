"""T-LESS verifier test: draw K LiePose samples per test crop and compare four ways to pick one:
random (= LiePose), medoid (voting), min world-model energy, oracle best.

Poses stay in LiePose's own parametrisation (allocentric rotation, scaled translation). Rotation error is
the min angle to the GT's symmetry set (models_info symmetries); translation error is L2 in scaled units.

  DATADIR=... CACHE_ROOT_PATH=... PYTHONPATH=src python scripts/tless/verifier_eval.py <tless_exp> --step 400000 \
      --wm $LOGS/probe/tless_v1 --tokens $DATADIR/tless_tokens --k 16 --out tless_verifier.json
"""

import argparse
import json
import os

import flax
import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

import liepose.exp.bop.dataset_registry  # noqa: F401
from liepose.exp.bop import run as bop_run
from liepose.exp.jepa_probe.probe import pose_feat
from liepose.exp.jepa_probe.probe_frozen import FG_FLOOR, weighted_energy
from liepose.exp.jepa_probe.probe_tless import TemplatePoseModel, load_symmetries
from liepose.utils.exploration import Options


def main():
  p = argparse.ArgumentParser()
  p.add_argument("exp")
  p.add_argument("--step", type=int, default=400000)
  p.add_argument("--wm", required=True)
  p.add_argument("--tokens", required=True)
  p.add_argument("--k", type=int, default=16)
  p.add_argument("--max_batches", type=int, default=None)
  p.add_argument("--batch", type=int, default=32)
  p.add_argument("--out", required=True)
  p.add_argument("--no_ema", action="store_true", help="sample with the raw training params instead of the EMA params")
  c = p.parse_args()

  a = Options(OmegaConf.to_container(OmegaConf.load(os.path.join(c.exp, "configs.yaml"))))
  a.pop("sample_options", None)
  a.restore_path, a.restore_step = os.path.join(c.exp, "ckpt"), c.step
  if c.no_ema:
    a.ema_enabled = False
  tb = bop_run.get_testbed("se3")(a, c.exp, setup=True)
  tb.opt = Options(use_ddpm=False, eta=1.0, beta_skip=True, steps=100, batch_size=c.batch, n_slices=c.k)
  lm, ns = tb.lie_metrics, tb.noise_schedule
  get_batch = tb.create_get_flat_batch_test_fn()
  backbone = jax.jit(tb.create_backbone_apply())
  p_sample = jax.jit(tb.create_p_sample_apply())
  T = ns.timesteps
  cur = (np.linspace(T, 0, 100, endpoint=False) - 1).astype(np.int32).tolist()
  seq = list(zip(cur, cur[1:] + [0]))

  @jax.jit
  def to_rt(r):
    g = jax.vmap(lm.as_lie)(r)
    return jax.vmap(lambda x: x.rotation().as_matrix())(g), jax.vmap(lambda x: x.translation())(g)

  # world model
  wa = json.load(open(os.path.join(c.wm, "args.json")))
  objs = np.load(os.path.join(c.wm, "objects.npy"))
  obj2idx = {int(o): i for i, o in enumerate(objs)}
  tok_all = np.load(os.path.join(c.tokens, "test_tokens.npy"), mmap_mode="r")
  fg_all = np.load(os.path.join(c.tokens, "test_fg.npy"), mmap_mode="r")
  row_of = {int(ix): r for r, ix in enumerate(np.load(os.path.join(c.tokens, "test_meta.npz"))["index"])}
  wm = TemplatePoseModel(n_obj=len(objs), n_tok=tok_all.shape[1], tok_dim=tok_all.shape[2], dim=wa["dim"], depth=wa["depth"])
  init = wm.init(jax.random.PRNGKey(0), jnp.zeros((1,), jnp.int32), jnp.zeros((1, 9)))
  wm_params = flax.serialization.from_bytes(init, open(os.path.join(c.wm, "params.msgpack"), "rb").read())
  wm_params = jax.tree_util.tree_map(jnp.asarray, wm_params)  # restored leaves are numpy; templates are indexed by a tracer

  @jax.jit
  def energies(obj, g, tok, fg):
    pred = wm.apply(wm_params, obj, g)
    return weighted_energy(pred, tok.astype(jnp.float32), FG_FLOOR + fg)

  syms = load_symmetries(os.path.join(os.environ["DATADIR"], "bop_datasets/tless/models_eval/models_info.json"))

  def rot_err(R, R_gt, S):
    eq = R_gt @ S  # (M,3,3)
    tr = np.einsum("kij,mij->km", R, eq)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).min(1)

  test = tb.create_test_dataset()
  key = jax.random.PRNGKey(11)
  recs = {k: [] for k in ["random", "medoid_t", "min_energy", "oracle"]}
  trecs = {k: [] for k in recs}
  allS = {k: [] for k in ["R", "t", "energy", "rot_err", "tran_err", "index", "R_gt", "t_gt"]}  # every sample, for scorer/refiner
  objs_out = []
  for bi, data in enumerate(tqdm(test, desc="sample")):
    if c.max_batches is not None and bi >= c.max_batches:
      break
    batch = get_batch(jax.random.PRNGKey(bi), data, n_slices=c.k)
    feat = backbone(batch["img"])
    rt = batch["rt"]
    for t, tp in seq:
      key, k1 = jax.random.split(key)
      _, rt = p_sample(k1, feat, rt, t, tp)
    R, tr_ = map(np.asarray, to_rt(rt))
    B = len(data["object_id"])
    R, tr_ = R.reshape(B, c.k, 3, 3), tr_.reshape(B, c.k, 3)
    for i in range(B):
      oid = int(data["object_id"][i])
      R_gt = np.asarray(jaxlie.SO3(jnp.asarray(data["rotation_allo"][i])).as_matrix())
      t_gt = np.asarray(data["translation_scaled"][i])
      re = rot_err(R[i], R_gt, syms[oid])
      te = np.linalg.norm(tr_[i] - t_gt, axis=-1)
      r = row_of[int(data["index"][i])]
      e = np.asarray(energies(jnp.full((c.k,), obj2idx[oid]), pose_feat(R[i], tr_[i]),
                              jnp.broadcast_to(jnp.asarray(tok_all[r]), (c.k,) + tok_all.shape[1:]),
                              jnp.broadcast_to(jnp.asarray(fg_all[r], jnp.float32), (c.k,) + fg_all.shape[1:])))
      dt = np.linalg.norm(tr_[i][:, None] - tr_[i][None], axis=-1)
      picks = {"random": 0, "medoid_t": int(dt.sum(1).argmin()), "min_energy": int(e.argmin()),
               "oracle": int(np.argmin(re / 10.0 + te / 0.05))}
      for k_, j in picks.items():
        recs[k_].append(re[j])
        trecs[k_].append(te[j])
      objs_out.append(oid)
      for k_, v_ in [("R", R[i]), ("t", tr_[i]), ("energy", e), ("rot_err", re), ("tran_err", te),
                     ("index", int(data["index"][i])), ("R_gt", R_gt), ("t_gt", t_gt)]:
        allS[k_].append(v_)

  objs_out = np.array(objs_out)
  out = {"n": int(len(objs_out)), "k": c.k}
  print(f"T-LESS test crops: {len(objs_out)}, K={c.k}")
  print(f"{'pick':12s} {'R mean':>7s} {'R med':>6s} {'R<5':>6s} {'R<10':>6s} {'>30deg':>7s} {'t mean':>7s} {'t med':>6s}")
  for k_ in recs:
    r, t = np.array(recs[k_]), np.array(trecs[k_])
    out[k_] = {"rot_mean": float(r.mean()), "rot_median": float(np.median(r)), "rot_lt5": float(100 * (r < 5).mean()),
               "rot_lt10": float(100 * (r < 10).mean()), "rot_gt30": float(100 * (r > 30).mean()),
               "tran_mean": float(t.mean()), "tran_median": float(np.median(t))}
    v = out[k_]
    print(f"{k_:12s} {v['rot_mean']:7.2f} {v['rot_median']:6.2f} {v['rot_lt5']:6.1f} {v['rot_lt10']:6.1f} {v['rot_gt30']:7.2f} "
          f"{v['tran_mean']:7.4f} {v['tran_median']:6.4f}")
  np.savez(c.out.replace(".json", "_samples.npz"), obj=objs_out, k=c.k, **{k: np.stack(v) for k, v in allS.items()})
  np.savez(c.out.replace(".json", ".npz"), obj=objs_out, **{f"rot_{k}": np.array(v) for k, v in recs.items()},
           **{f"tran_{k}": np.array(v) for k, v in trecs.items()})
  json.dump(out, open(c.out, "w"), indent=2)


if __name__ == "__main__":
  main()
