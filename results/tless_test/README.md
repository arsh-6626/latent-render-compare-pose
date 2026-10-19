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

Reproduction: the HPC checkpoint's EMA weights (LiePose's test loop samples with them when ema_enabled) give a
training loss of 5.39 vs 2.50 for the live weights (the logged training loss), although they differ from the live
weights by only ~0.2% (relative norm); sampling with the live weights (scripts/tless/test_liepose.py --no_ema)
reproduces the paper. Deviation to disclose: live instead of EMA weights. Our own sp_dit/tless_dit models are not
affected (B: EMA 1.254 vs live 1.277 training loss).
