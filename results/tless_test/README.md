# T-LESS results (BOP19 targets, GT visible box + visible mask crops, LiePose's TestDataset + Evaluator)

VSD uses bop_toolkit's vispy renderer (bop_renderer not installed); CSVs of the two lrc runs were produced on HPC
dtu004 and scored locally with the same Evaluator.evaluate (HPC scoring failed: system python on PATH).

| run | AR | MSPD | MSSD | VSD |
|---|---|---|---|---|
| LiePose SE(3), published (Hsiao et al. CVPR'24, Table 4) | 0.701 | 93.16 | 60.17 | 56.88 |
| liepose_se3_400k (our HPC-trained reproduction) | 0.093 | 26.81 | 0.57 | 0.39 |
| lrc_dit_nowm_tless_ret_v1_100k (control, no WM) | 0.412 | 75.59 | 25.80 | 22.23 |
| lrc_dit_tless_ret_v1_100k (latent render-and-compare) | 0.547 | 83.30 | 42.52 | 38.20 |

liepose_se3_400k is NOT a valid baseline: it is far below the published numbers (rotation as well as depth), i.e.
the reproduction is broken (cause not yet found). Compare against the published LiePose numbers until it is fixed.
