# Project brief for the explainer video (read first; MASTER_PROMPT.md is the full specification)

You are a separate Claude session making a 3Blue1Brown-style explainer video of our own CVPR submission. The full spec is
`video/MASTER_PROMPT.md`; this brief fills its placeholders and adds hard project constraints.

## Placeholders

- {{PAPER_TITLE}}: Latent Render-and-Compare: SE(3) Diffusion with a Feature-Field World Model
- {{METHOD_NAME}}: LaRC (latex macro `\method`), a latent render-and-compare feature-field world model inside SE(3) diffusion
  (LiePose).
- {{BASELINE(S)}}: (1) LiePose, the SE(3) diffusion pose estimator of Hsiao et al. 2024 (ResNet-34, no world model), and (2) our
  no-world-model control: the identical DINOv2-S denoiser with the world model removed (`den_mode=field_nowm`).
- {{OUTPUT_PATH}}: /home/kavinder/ARNABI_ARSH/LARp/video/out/larc_explainer.mp4
- {{HERO_VISUALIZATION}}: decide from the paper (sec/4_method.tex). Suggested: the render-and-compare energy over SE(3). A
  candidate pose (R, t) is rendered by the feature field into the 8x8 DINOv2 token space of the crop and compared with the
  observed tokens; show this energy as a landscape over rotations/translations, and how the denoiser's reverse diffusion
  trajectory on SE(3) uses it. Build it from the actual definitions in the paper; do not invent components.
- {{HEADLINE_RESULTS}} / {{BASELINE_NUMBERS}}: use ONLY numbers that are written in the paper source
  (`paper/sec/5_experiments.tex`, `paper/sec/X_suppl.tex`), never `\pend` / `\TODO` cells. As of 2026-10-31:
  - T-LESS (BOP19, GT crops, AR %): LaRC 100k x 32: VSD 65.3 / MSSD 65.5 / MSPD 86.5 / AR 72.4; control 21.5 / 24.9 / 75.2 /
    40.5; LiePose published 56.9 / 60.2 / 93.2 / 70.1 (AR = mean, computed by us); LiePose trained by us at 400k x 64: AR 73.7.
  - SPEED+ (ESA score, lower is better, lightbox / sunlamp, 400k x 64): LaRC 0.672 / 0.664 (16-sample consensus 0.565 / 0.568);
    control 0.854 / 0.793 (0.776 / 0.688); LiePose trained by us 0.234 / 0.289 (0.165 / 0.223). Published SOTA is far
    better (e.g. SPNv3-B 0.051 / 0.079).
  - Re-read the paper before rendering the results scene in case numbers changed; the paper is the source of truth.
- {{QUALITATIVE_RESULTS}}: only real data. Crops / poses must come from the actual datasets and our actual outputs (ask in
  the session if you need files; do not fabricate examples). If no qualitative outputs are available, show the mathematics on
  real CAD models (T-LESS models: /media/kavinder/hdd2/liepose_data/bop_datasets/tless/models_cad, if present).

## Honesty constraints (hard)

- The main claim is relative: the world model gives large gains over the identical denoiser without it (T-LESS +31.9 AR,
  SPEED+ score 0.85 -> 0.67 lightbox). It does NOT beat the state of the art: on T-LESS it beats the published LiePose GT-crop
  number but is 1.3 AR below LiePose trained by us at 8x the training images; detector-crop methods (GDRNPP 78.6) are a
  different protocol; on SPEED+ it is far from the SOTA and from LiePose. Say this in the results scene.
- GT-crop vs detector-crop rows are not comparable; never present them as a head-to-head.
- Do not show YCB-V results (runs in progress), selection, or any `\pend` row.
- The retrieval world model is a supplement variant, not the main method.

## Environment constraints (hard)

- Work only inside /home/kavinder/ARNABI_ARSH/LARp/video/. Read the paper and code (LARp/paper, LARp/src) but do not edit them.
- Do NOT touch running processes, GPUs in use, HPC jobs (no ssh to iitd-hpc08 / 10.141.90.34), or other folders.
  Rendering is CPU work; don't use GPU 0 heavily (a cache build and other users' jobs use it).
- No external transfers of the paper's content: it is unpublished. Use a LOCAL text-to-speech engine only (e.g. Kokoro or
  Piper installed with pip into a new conda env); do NOT use gTTS, ElevenLabs, OpenAI or any cloud TTS, and do not upload
  frames or text anywhere. Installing packages from PyPI / GitHub is fine.
- Framework: 3b1b/manim (manimgl). The existing `manim` conda env is Manim Community 0.22 — not the required framework;
  create a new env (e.g. `conda create -n manimgl python=3.11`) and `pip install manimgl`. LaTeX and ffmpeg are installed.
- Disk: / has ~50 GB free, /media/kavinder/hdd2 ~100 GB (do not fill it); put large intermediate renders in
  /media/kavinder/hdd3/larc_video/ (6.9 TB free) and only the final MP4 in video/out/.
- Do not git commit or push.

When done, write video/out/REPORT.md: duration, scene list, which numbers were used and from which file/line, and any
known issues.

## Workflow with the user (hard)

1. Read the paper (paper/sec/*.tex, including X_suppl.tex) and MASTER_PROMPT.md first.
2. BEFORE writing any storyboard, narration or code: propose the video's structure to the user in this session — the scene
   list with headings, what each scene explains and shows (the visual), its equations, and an estimated duration — and
   discuss it. Do not start production until the user approves the structure.
3. Pacing: never slow the video down or pad it to reach the length. If it is short, add more explanatory content
   (more derivation, more of the method's actual mechanics, more of the real experiments and ablations), not slower
   animations or longer pauses.

## In-progress: flow matching variant (added 2026-10-31)

We are adding a flow-matching version of the generative process as a supplement experiment (T-LESS, LaRC and control,
100k x 32; NO results yet, not in the paper yet). Same world model, tokens, denoiser and selection; only the process changes:
- path: geodesic on SE(3) between the data pose g_0 and a prior pose g_1 (uniform rotation, uniform translation in [-1,1]^3,
  the same prior as the diffusion sampler): g_tau = g_0 Exp(tau xi), xi = Log(g_0^{-1} g_1), tau in [0,1];
- the network predicts the constant body-frame velocity xi (6-d, se(3)); loss ||v_theta - xi||^2;
- sampling: Euler ODE on the group from tau=1 to 0, g <- g Exp(-dtau v_theta), 10-20 steps instead of 100; every step still
  renders the field at the current g and compares it with the observation.
Motivation: the world model is evaluated once per step and sample, so fewer steps cut test cost; and "which direction to
move the pose so the rendering matches" is the render-and-compare reading of the velocity.
Video: you may include it as a short, clearly labelled "ongoing / variant" segment (the maths of diffusion vs flow on SE(3)
is a good visual), but never show or imply results for it unless they appear in the paper source. Discuss with the user
whether to include it when you propose the structure.

## APPROVED STRUCTURE (user, 2026-10-31) — the structure step is done; go straight to production

Decisions: include the flow-matching segment (clearly labelled "ongoing variant, no results"); local offline TTS is OK
(Kokoro preferred, Piper fallback); skip the retrieval world model; target ~14 min with dense content, no slow pacing.
Qualitative material: real T-LESS CAD models / real crops from /media/kavinder/hdd2/liepose_data/bop_datasets/tless (read
only); you have no sampler outputs, so trajectories you animate are mathematical demonstrations (label them as
illustrations of the procedure, never as model outputs or results).

```
#   Heading                                   min   Content                                              Visual
1   The pose problem                          1.0   6D pose from one crop; symmetry/occlusion -> multi-modal  real T-LESS object rotating; its crop; symmetric orbit on SO(3)
2   Poses as a group                          1.5   SE(3), allocentric R, t=(dx/w,dy/h,z/r), Exp/Log, se(3)   crop-relative frame; tangent plane + Exp onto the group
3   Diffusion on SE(3) (LiePose)              1.5   g_t = g0 Exp(sigma_t eps), VE power-law, eps-pred, 100 DDIM pose cloud diffusing in the axis-angle ball (antipodal id.), reverse
4   What the denoiser never checks            1.0   eps_theta(g_t,t,I): image once as a vector                 same image, two hypotheses: input differs only in a 6-vector
5   Render-and-compare, and why it's costly   0.8   CAD + renderer, separate refinement stage                 mesh rendered at a hypothesis, overlay, residual
6   The tokens we compare in                  1.0   DINOv2 ViT-S/14, 16x16 -> 8x8, z_obs in R^{64x384}, w_p, k  crop -> patches -> tokens -> pooled grid; mask dilation
7   HERO: the feature field and its energy    3.0   tri-planes + MLP -> sigma, h; 16x16 rays x 32 samples;    3D box, rays, samples, tri-plane projection, tokens on grid;
                                                    volume rendering, opacity, ray latent, appearance code;   energy landscape over a slice of SO(3), minima at true pose
                                                    E(g; z_obs); exact in the pose                             and symmetric copies
8   Fitting the field                         0.8   reconstruction only at GT poses; density unsupervised      training views around object; density emerging, no mesh
9   Comparing inside every step               1.5   x_p = Lin[LN z_obs, LN z_hat, LN(z_hat - z_obs), w, LN h]  reverse trajectory with the rendering at g_t and the shrinking
                                                    + e_p; conditioning c; 4 adaLN blocks; stop-gradient;      difference grid; side by side with the control on the same
                                                    control removes only the rendered quantities              prior sample
10  Selecting among samples                   0.7   N=16, argmin E; consensus rule                             16 trajectories into different modes, ranked by energy
11  Flow instead of diffusion (ongoing)       1.0   g_tau = g0 Exp(tau xi), velocity xi, Euler on the group    pose cloud flowing along geodesics vs diffusing; "in progress"
12  Results                                   1.5   paper numbers only (see Placeholders); protocol caveats    animated bars from the paper's numbers; GT vs detector groups
                                                                                                              separate; SPEED+ on a log axis with published SOTA
13  What the world model can and can't see    0.8   P(E>E_gt) by perturbation size (tab:wm-disc); failures    plot by perturbation size; symmetric tie
14  Synthesis                                 0.7   back to the energy; claim and limits                       energy landscape, trajectory descending
```
