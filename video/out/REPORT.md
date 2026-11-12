# LaRC explainer video: report

**File:** `video/out/larc_explainer.mp4`. H.264 (yuv420p, CRF 18), 1920x1080, 30 fps, AAC 48 kHz mono, 192 kb/s,
69.6 MB. **Duration: 14:56.6 (896.6 s).** Loudness is normalised to -16 LUFS: mean volume -16.6 dB, peak -1.3 dB. There
is no silence longer than 3 s.

## How it was made
- **Animation:** 3b1b/manim (manimgl 1.7.2), with all scenes in `video/scenes/`. The env is a new conda env at
  `/media/kavinder/hdd3/larc_video/envs/manimgl`; the `manim` env was not touched. Rendering ran headless through EGL on
  the Intel integrated GPU (`scenes/run_manim.py`). No NVIDIA GPU was used for rendering.
- **TeX:** the machine has no `dvisvgm`, so `run_manim.py` replaces manimgl's TeX-to-SVG step with `tectonic` +
  `pdftocairo`. Tectonic is the engine the paper's `build.sh` already uses.
- **Voice:** Kokoro-82M (`af_heart`, speed 1.0), run locally on CPU (`tools/tts.py`). Only the model weights were
  downloaded; no text or frames were uploaded. Pronunciation was checked with a local faster-whisper `base.en`.
- **Assembly** (`tools/assemble.py`): each narration beat is placed at the time its scene logged for it. Scenes wait for
  the previous beat's audio, so beats never overlap and no beat runs past its scene; the assembler checked both.
- **Script and storyboard:** `video/script.md` and `video/storyboard.md`.
- **Intermediates:** under `/media/kavinder/hdd3/larc_video/` (data/, audio/, render/, frames/, ckpt copies).
- **Scope:** nothing outside `LARp/video/` was edited. There was no git, ssh or HPC activity, and no running process was
  touched. Real-data extraction (`tools/real_data*.py`) ran in the existing liepose env, read-only, with
  JAX_PLATFORMS=cpu.

## Scenes and timings
| # | scene | start | length |
|---|---|---|---|
| S01 | The pose problem | 0:00.0 | 44.4 s |
| S02 | Poses as a group | 0:44.4 | 64.0 s |
| S03 | Diffusion on SE(3) (LiePose) | 1:48.4 | 71.0 s |
| S04 | What the denoiser never checks | 2:59.4 | 38.8 s |
| S05 | Render-and-compare | 3:38.2 | 44.1 s |
| S06 | The tokens we compare in | 4:22.3 | 61.0 s |
| S07 | HERO: the feature field and its energy | 5:23.3 | 179.4 s |
| S08 | Fitting the field | 8:22.7 | 37.1 s |
| S09 | Comparing inside every step | 8:59.7 | 80.7 s |
| S10 | Selecting among samples | 10:20.4 | 30.5 s |
| S11 | Flow instead of diffusion (in progress) | 10:50.9 | 58.9 s |
| S12 | Results | 11:49.8 | 88.2 s |
| S13 | What the world model can and cannot see | 13:18.0 | 51.1 s |
| S14 | Synthesis | 14:09.2 | 47.5 s |

## Every number in the video and its source
E = `paper/sec/5_experiments.tex`, S = `paper/sec/X_suppl.tex`, M = `paper/sec/4_method.tex`,
B = `paper/sec/3_background.tex`. No `\pend` cell is shown.

| number(s) | meaning | source |
|---|---|---|
| 21.5 / 24.9 / 75.2 / 40.5 | control VSD/MSSD/MSPD/AR, T-LESS, GT crops, 100k x 32 | E:72 |
| 65.3 / 65.5 / 86.5 / 72.4 | LaRC, same setting | E:73 |
| 40.5 -> 72.4, all three recalls rise | narration | E:145-146 |
| 26.8 -> 80.3 (translation < 2 cm) | control -> LaRC | E:146; S:275-276 |
| 45.7 -> 71.6 (rotation < 5 deg) | control -> LaRC | E:147; S:275-276 |
| 70.1 (dagger, mean computed by the authors) | LiePose published, GT crops | E:63, E:40 |
| 73.7 | LiePose trained by us, 400k x 64 | E:66 |
| "1.3 above LaRC", "an eighth of the training images" | budget statement | E:147-148, E:26, E:71 |
| 78.6 | GDRNPP, detector crops; shown behind a separator as "different protocol, not comparable" | E:60 |
| 0.854 / 0.793 | SPEED+ control, lightbox / sunlamp, 400k x 64 | E:116 |
| 0.672 / 0.664 | SPEED+ LaRC, 400k x 64 | E:118 |
| 0.234 / 0.289 | SPEED+ LiePose (our training) | E:114 |
| 0.051 / 0.079 | SPEED+ SPNv3-B, published, GT box | E:105 |
| 0.65 / 0.84 / 0.98; 84 %, 98 %; 0.52 below 1 deg | SPEED+ field discrimination | E:247, E:256-257 |
| 0.53 / 0.66 / 0.86 | T-LESS field discrimination | E:245 |
| median 3.4 deg, mean 14.2 deg | pick error of the field's energy, T-LESS | E:260; S:340 |
| 500 held-out training crops / 300 validation crops | discrimination protocol (caption) | E:234-235 |
| 224x224, 14-px patches, 16x16, 384, 8x8, 64x384 | observation tokens | M:26-29 (224 at 26, 384 at 28) |
| w_p = w_0 + m_p, w_0 = 0.05 | token weights | M:30 |
| r_o = 1.2 x object radius | the only CAD-derived quantity | M:62 |
| 3 planes 128x128, 32 ch; MLP width 128; h in R^128 | field | M:38-40; E:31 |
| 16x16 rays, 32 samples within +-r_o | rendering | M:44-45 |
| D = 384 | energy | M:71 (eq:energy) |
| 180,224 training crops | T-LESS cache | S:21 |
| 100k fitting steps | field fit, T-LESS | M:77 |
| 3*384+1+128 = 1281 | denoiser input dims (sum of the parts listed) | S:31 |
| 4 adaLN blocks, width 256 | denoiser | M:95; E:32 |
| N = 16; 10 deg consensus | selection | M:118-119 |
| T = 100, VE power law; sigma from 1e-4 to 1 | schedule; the curve is computed with LiePose's code (start variance 1e-8, power 3) | B:13; `src/liepose/noise/ve/power.py`, `exp/bop/run.py` |
| 10-20 Euler steps | flow-matching variant, planned; labelled "in progress, no results" | BRIEF.md (not in the paper) |

**Constants on screen that are not paper results:**
- 720x540 image plane: T-LESS Primesense camera.
- 180 deg symmetry: BOP models_info of object 5.
- 40 deg wrong pose: chosen for the illustration.
- 40^3 grid / top 12% density and 320-point noise cloud: visualisation parameters.
- sigma endpoints 1e-4 and 1: LiePose code, as above.
- 10-20 Euler steps: BRIEF.md.

The sampler-run statistics in the provenance section below are not in the paper and are not shown in the video.

**Not shown:** YCB-V, the retrieval world model, the selection table (all `\pend`), runtime (all `\pend`), the
LaRC + selection rows, the trained-encoder rows, and any flow-matching result. No state-of-the-art claim is made; the
results scene ends with "not a new state of the art". Detector-crop rows are kept separate from GT-crop rows.

## Real data and model outputs used (provenance)
- **Crop:** T-LESS test crop, object 5, cache row 211, chosen by a fixed rule: the first object-5 test crop in cache order
  with at least 25% visible mask.
- **Field-only quantities, shown as the mathematics on real data:** renders at the GT pose and 40 deg off, PCA token
  grids, the learned tri-planes, the density (top 12% of a 40^3 grid), per-ray weights, and the energy landscape on a
  2-D slice of SO(3). They come from `tl_field_v1`. No energy value is shown as a number.
- **Sampler runs** (S03, S09, S10, S11 left, S14): LaRC and control, 100k x 32 checkpoints, EMA weights, 16 samples from
  the same priors with the same noise. They are one test crop and captioned on screen as "real run ... an example, not a
  result". In this run 15/16 LaRC samples ended within about 3.3 deg (symmetric) and one about 107 deg off, which had the
  highest energy. For the control, 15/16 ended within about 4.6 deg and one about 94 deg off. These figures are not shown
  in the video.
- **S11 flow paths:** geodesic interpolants g0 Exp(tau xi) from the run's priors. They are mathematics, not a trained
  model, and are captioned that way.
- **S04 "ResNet-34 features":** a schematic vector, labelled "schematic".

## Mathematics verification (`tools/verify_math.py`, all pass)
SE(3) Exp/Log/composition vs jaxlie (<= 5e-7); sigma_t vs LiePose's PowerNoiseSchedule (3e-8); forward process vs
jaxlie; geodesic endpoints; trajectory tangent coordinates -> poses; volume weights recomputed from sigma (1e-7); pooled
opacity = the field's own output (8e-7); energy formula = the field run's energy (1e-6); Haar angle density.

## Known issues
- **Frame rate:** 30 fps (the minimum allowed; 60 is preferred). Rendering at 60 fps is a flag change (`--fps 60`
  needs a config file because of a CLI bug) and roughly doubles render time.
- **Text size:** some small captions (24-26 pt at 1080p) are readable but small.
- **Image of the object:** S02's delta_x / delta_y offsets are tiny, which is real: the object centre is near the crop
  centre, so the labels crowd the inset.
- **Learned tri-planes:** they look noisy. That is how the trained parameters look (init noise plus learned structure).
- **Learned density:** its shape follows the object's extent and is box-like, not a sharp surface. That is real, since
  the field is soft; the comparison with the CAD model is qualitative.
- **Weights grid:** the token cache stores fractional foreground weights. The real cached values are shown, with the
  paper's formula, which defines m_p as binary.
- **Narration length:** the narration fills about 14.4 min of the 14.9 min. Kokoro reads "Fourier" a little oddly.
- **Not done:** no qualitative LiePose outputs (no LiePose run was made); no flow-matching results (none exist).
- **Report file:** the Write tool refused report files for subagents, so this file was written via the shell because
  the brief requests it.
