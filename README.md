# LARp: latent render-and-compare pose diffusion

SE(3) diffusion (LiePose) with a frozen world model in the loop: at every denoising step each pose hypothesis is
rendered in DINOv2 token space and compared with the observation's tokens; the comparison conditions the denoiser.
No CAD model. Ported from `../liepose-diffusion` (2026-10-27) keeping the working pipeline only; the failed branches
(world models v2-v4, the WM verifier, the flip-fix pilot, the joint JEPA model, MPC planning, the T-LESS refiner)
stay in the original repo. `results/` keeps every result, including the failed experiments' diagnostics, since the
analysis relies on them.

## Layout

```
src/liepose/
  dist/ metrics/ noise/ utils/        LiePose: SE(3) / SO(3) diffusion, metrics, noise schedules
  data/                               BOP (T-LESS), SPEED+, SYMSOL, SYMSOL-T datasets + evaluators
  exp/testbed.py, exp/model.py        LiePose training / sampling loop and denoiser
  exp/{bop,speedplus,symsol,symsolt}  LiePose baselines per dataset
  exp/cmp/                            latent render-and-compare (LaRC)
    dit.py, dit_ret.py                token denoiser (RetDenoiser), modes dit / dit_nowm (control) + ablations
    run.py                            SYMSOL-T      sp_dit.py   SPEED+ (dit_ft: trained DINOv2 backbone)
    tless_dit.py                      T-LESS (train_objs: held-out-object / zero-shot test)
  exp/jepa_probe/
    probe_ret.py, probe_ret_tless.py  retrieval world models used in the loop (SYMSOL-T v7c, T-LESS / SPEED+ v1)
    sp_geometry.py                    SPEED+ camera / crop geometry
    field.py                          3D feature-field world model (no retrieval; --lat_dim: per-image appearance)
    wm_eval.py                        world-model evaluation harness (P(E_pert > E_gt) per bin, refiner protocol)
  models/dinov2.py                    DINOv2 ViT-S/14 in flax (scripts/speedplus/convert_dinov2.py)
scripts/
  speedplus/   export_crops, augment_crops, test_liepose, esa_score (official ESA), convert_dinov2,
               field_diag / field_fit_eval / field_refine (feature-field diagnostics + energy refinement),
               tok_alpha_oracle / tok_sens_inplane (what frozen tokens can resolve)
  tless/       export_crops, encode_crops, test_liepose, eval_csv (BOP19 AR), score_subsets (seen / unseen AR)
  oracle/      crop_encode, cache_dino, make_holdout (SYMSOL-T token caches and WM selection split)
  hpc/         IITD scai_q job scripts for the current runs + resume_loop.sh
  launch/      local launchers
lib/           bop_toolkit, bop_renderer
results/       committed result JSONs / logs (benchmark tables only from the official protocols)
paper/
```

## Environment

`liepose` conda env (JAX 0.6, flax 0.10); the token encoders (`encode_crops`, `crop_encode`, `convert_dinov2`)
run in the torch env (`jepa`). Data: `DATADIR` (token caches, crops, BOP datasets), `SPEEDPLUS_ROOT`.

```
export DATADIR=/media/kavinder/hdd2/liepose_data SPEEDPLUS_ROOT=$DATADIR/speedplusv2_ann
export PYTHONPATH=src:lib/bop_toolkit
python -m liepose.exp.cmp.sp_dit --root_path $LOGS/speedplus-lrc "den_mode=[dit]" "wm_dir=[$DATADIR/logs/probe/sp_geo_v1]"
python -m liepose.exp.cmp.tless_dit --root_path $LOGS/tless-lrc "den_mode=[dit]" "wm_dir=[$DATADIR/logs/probe/tless_ret_v1]"
python -m liepose.exp.jepa_probe.field --out $DATADIR/logs/probe/sp_field_v1
```
