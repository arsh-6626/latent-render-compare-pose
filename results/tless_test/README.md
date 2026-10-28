# T-LESS results (BOP19 targets, GT visible box + visible mask crops, LiePose's TestDataset + Evaluator)

VSD uses bop_toolkit's vispy renderer (bop_renderer not installed); CSVs of the two lrc runs were produced on HPC
dtu004 and scored locally with the same Evaluator.evaluate (HPC scoring failed: system python on PATH).

| run | AR | MSPD | MSSD | VSD |
|---|---|---|---|---|
| LiePose SE(3), published (Hsiao et al. CVPR'24, Table 4) | 0.701 | 93.16 | 60.17 | 56.88 |
| liepose_se3_400k_noema (our reproduction, live weights) | 0.737 | 92.33 | 65.72 | 63.12 |
| liepose_se3_400k (our reproduction, EMA weights) - INVALID | 0.093 | 26.81 | 0.57 | 0.39 |
| lrc_dit_nowm_tless_ret_v1_100k (control, no WM) | 0.412 | 75.59 | 25.80 | 22.23 |
| lrc_dit_tless_ret_v1_100k (latent render-and-compare) | 0.547 | 83.30 | 42.52 | 38.20 |
| lrc_dit_nowm_tless_ret_v1_100k_s43 (control, seed 43) | 0.425 | 76.88 | 27.09 | 23.39 |
| lrc_dit_tless_ret_v1_100k_s43 (latent render-and-compare, seed 43) | 0.541 | 82.57 | 41.96 | 37.82 |
| lrc_dit_in2_tless_ret_v1_100k (input: [z_obs, ẑ]) | 0.537 | 82.50 | 41.62 | 37.12 |
| lrc_dit_in3_tless_ret_v1_100k (input: [z_obs, ẑ, ẑ - z_obs]) | 0.543 | 83.00 | 42.14 | 37.84 |
| lrc_dit_hi_tless_ret_v1_100k (WM only at t >= T/2, high noise) | 0.403 | 77.27 | 23.61 | 20.07 |
| lrc_dit_lo_tless_ret_v1_100k (WM only at t < T/2, low noise) | 0.524 | 80.16 | 40.74 | 36.31 |

Ablations (seed 42 unless noted; frozen DINOv2, 100k, same test protocol; AR = mean of the per-object averaged
MSPD / MSSD / VSD, as above). Two seeds: control 0.412 / 0.425, world model 0.547 / 0.541 (gain +0.135 / +0.116, mean +0.126). The rendered tokens carry the gain
(+0.125 of +0.135); the explicit difference and the weight token add +0.006 / +0.004, within the seed spread. The
world model contributes at low noise (0.524, ~85% of the gain) and not at high noise (0.403, no better than the
control); both halves together give the full 0.547 (high-noise steps add mostly MSPD).


Reproduction: the HPC checkpoint's EMA weights (LiePose's test loop samples with them when ema_enabled) give a
training loss of 5.39 vs 2.50 for the live weights (the logged training loss), although they differ from the live
weights by only ~0.2% (relative norm); sampling with the live weights (scripts/tless/test_liepose.py --no_ema)
reproduces the paper. Deviation to disclose: live instead of EMA weights. Our own sp_dit/tless_dit models are not
affected (B: EMA 1.254 vs live 1.277 training loss).
