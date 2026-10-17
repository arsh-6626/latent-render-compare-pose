# Latent render-and-compare 6D pose diffusion

Work in progress (target: CVPR 2027). A JEPA world model on frozen DINOv2 tokens acts as the *renderer* inside
LiePose's SE(3) score-based diffusion: at every denoising step the model predicts the tokens the object would
produce at the current noisy pose hypothesis and the denoiser compares them with the observation.

Built on [LiePose](https://github.com/Ending2015a/liepose-diffusion) (Hsiao et al., CVPR 2024, MIT licence);
the original README is in [`docs/LIEPOSE_README.md`](docs/LIEPOSE_README.md).

## Layout

```
src/liepose/
  exp/symsolt, symsol, bop     upstream LiePose experiments (SYMSOL-T, SYMSOL, BOP/T-LESS)
  exp/cmp/                     compare-in-the-loop denoisers
    dit_ret.py                 latent render-and-compare denoiser (modes: dit, dit_nowm, dit_ray)
    run.py, testbed.py         training / eval entry point (SE(3) diffusion)
    bop_run.py                 T-LESS variant
  exp/jepa_probe/              world models (token predictors) and their ceiling evals
    probe_frozen.py            template WM on cached DINOv2 tokens (SYMSOL-T)
    probe_ret.py               retrieval WM: reference crop + geometry tokens (model C = v7c_crop_geo_roll)
    probe_tless.py             template WM for T-LESS / SPEED+
    probe_ret_tless.py         retrieval WM for T-LESS / SPEED+ (val/test protocol, symmetric metrics)
  exp/jepa_liepose/            earlier JEPA-conditioned LiePose variant
  exp/speedplus/, data/speedplus/   SPEED+ spacecraft dataset and LiePose runner
  models/djepa_jax.py          D-JEPA encoder port
  planning/mpc_cem.py          CEM planning in the world model
scripts/
  oracle/                      DINOv2 caching, crop encoding, refinement-ceiling and WM-planning diagnostics
  tless/, speedplus/           crop/token export and evaluation for the real datasets
  wmden_select.py              K-sample selection (voting / WM energy / oracle)
  hpc/                         PBS scripts for the IITD HPC (scai_q)
  launch/                      exact local launch scripts used for the reported runs
results/                       small JSON/YAML outputs (args, ceiling.json, eval summaries) for every run
docs/                          upstream README
```

Checkpoints, token caches and datasets are not in the repo (they live under `$DATADIR`, e.g. `crop_cache/`,
`tless_tokens/`, `speedplus_tokens/`).

## Quick start

```bash
pip install -e .            # plus requirements.txt; JAX/flax as in the upstream README
export DATADIR=/path/to/liepose_data PYTHONPATH=src
# latent render-and-compare denoiser on SYMSOL-T with world model C
python -m liepose.exp.cmp.run --root_path $DATADIR/logs/experiments/symsolt-lrc "den_mode=[dit]" "wm_kind=[ret]" \
  "wm_dir=[$DATADIR/logs/probe/v7c_crop_geo_roll]" "cache_dir=[$DATADIR/crop_cache]" "train_steps=[100000]"
```

See `scripts/launch/*.sh` for every run's full command line.

## Current results (SYMSOL-T, 25k test images, 100k steps)

| denoiser | rot mean | rot median | tran mean |
|---|---|---|---|
| `dit_nowm` (no world model) | 1.742° | 1.148° | 0.0611 |
| `dit` (world model C), K=1 | 0.989° | 0.491° | 0.0295 |
| `dit` K=16 + voting | 0.551° | 0.476° | 0.0210 |
| `dit_ray`, K=1 | 0.965° | 0.492° | 0.0284 |

Per-run numbers are in `results/`.
