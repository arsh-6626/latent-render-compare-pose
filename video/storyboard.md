# Storyboard: Latent Render-and-Compare (LaRC) explainer

Approved structure (BRIEF.md), 14 scenes. Narration: `script.md` (beats `[Sxx.n]`). Code: `scenes/sNN_*.py` (manimgl 1.7.2,
headless EGL on the Intel iGPU via `scenes/run_manim.py`). Durations are the narration beats synthesised with Kokoro
(af_heart, speed 1.0) plus short gaps; each scene waits for its beat's audio, so picture and voice stay aligned.

## Real data behind the pictures (all computed read-only from the paper's code, data and checkpoints, CPU only)

| file (under /media/kavinder/hdd3/larc_video/data) | produced by | content |
|---|---|---|
| crop.png, mask.png, tokens.npz | tools/real_data.py | T-LESS test crop (object 5, first test crop in cache order with visible mask >= 25%: cache row 211), its cached DINOv2 tokens z (64x384), weights w, crop camera k, GT pose; field renders at the GT pose and at a pose 40 deg off (z_hat, opacity, latent); one PCA basis for all token colours |
| landscape.npz | tools/real_data.py | E(g; z) of the trained field (tl_field_v1) on a 73x37 slice of SO(3): rotation about the object's symmetry axis (-180..180 deg) x about a perpendicular axis (-90..90 deg), translation at GT |
| density.npz | tools/real_data.py | field density sigma(x) and latent h(x) (PCA) on a 40^3 grid in the object box |
| traj.npz | tools/real_data.py | real sampler runs (LiePose's 100-step sampler, eta = 1) of LaRC (`field_full_lat`) and the control (`field_nowm`), 100k x 32 checkpoints, EMA weights, 16 samples from the same priors with the same noise keys, every step; renders of sample 0 at every step; energies |
| rays.npz | tools/real_data2.py | per-ray volume-rendering quantities (sigma, alpha, T, w, opacity) of the 16x16 rays at the GT pose (bin-centre samples as at test time); CAD vertices projected through the crop camera |
| plane_*.png, train_views.npz | tools/real_data3.py | learned tri-planes of object 5 (PCA to RGB); viewing directions of the 6,010 training crops of object 5 |
| rc/*.png | tools/rc_frames.py | decimated CAD mesh projected with the crop camera along the geodesic wrong pose -> GT; residual = silhouette XOR the real visible mask |

Mathematics checked in `tools/verify_math.py` (all pass): SE(3) Exp/Log and composition vs jaxlie; sigma_t vs LiePose's
PowerNoiseSchedule; forward process vs jaxlie; geodesic endpoints; trajectory tangent coordinates -> poses; volume
weights recomputed from sigma; pooled opacity equals the field's own output; the energy formula equals the field run's
energies; Haar angle density.

## Colour code (fixed for the whole video)
observation z: light blue; world-model rendering z_hat and LaRC: orange; difference: pink; field latent h: purple;
hypothesis g_t / sample heads: yellow; ground truth and its symmetric copy: green; control: blue-grey; LiePose: indigo;
flow matching: teal; noise epsilon: red. Energy colour map: dark purple (low) -> yellow (high).

## Scenes

**S01 The pose problem (~43 s).** Real T-LESS crop (left); the CAD model of object 5 at the GT egocentric pose (right,
matches the crop), body axes, turntable. Symmetry axis (object z, from models_info) dashed; the model turns 180 deg about
it and looks the same ("same image"). Transition into the axis-angle ball of SO(3): GT and symmetric copy as two green
points; the orbit a rotationally symmetric object would have (a curve). Equation p(g | I), g in SE(3).

**S02 Poses as a group (~62 s).** Camera axes, translation arrow t, object at pose g with axes R; 4x4 matrix and composition
g1 g2. Image plane 720x540 with the real crop placed by its crop camera; magnified inset with crop centre and projected
object centre (delta_x, delta_y from the real GT translation); t = (dx/w, dy/h, z/r); allocentric R. Schematic group sphere
with tangent plane at e: vector omega grows, its Exp traced along a geodesic; on the right the real object turns about
omega by |omega| (Rodrigues). g' = g Exp(xi), xi = [v; omega], Exp(xi) = (Exp(omega), V(omega) v), xi = Log(g^-1 g').

**S03 Diffusion on SE(3) (~69 s).** g_t = g_0 Exp(sigma_t eps); six real standard-normal draws as bars; the object turns
as sigma grows (body-frame noise). The actual schedule sigma_t (power law, 1e-4 .. 1). Cloud of 320 noisy rotations of the
GT in the axis-angle ball as t goes 0 -> 100 (antipodal wrap visible). Training eps_theta(g_t, t, I) ~ eps and the
sampler's update as in LiePose's code (g0_hat = g_t Exp(sigma_t eps_hat)^-1, g_t' = g0_hat Exp(c eps_hat) Exp(s xi)). Real
reverse trajectories of the control denoiser (16 samples) contract onto the GT and its symmetric copy. Caption: real run,
control, 100k x 32, one test crop.

**S04 What the denoiser never checks (~38 s).** Crop -> feature column (labelled schematic), g_t as a 6-vector (real
hypotheses of the run), small MLP with activations flowing -> eps_hat. Two hypotheses (converged vs early sample, object
shown at each): the image features are identical, only the six numbers change. "would the object look like I at g_t?
never computed". Open loop.

**S05 Render-and-compare (~43 s).** Real crop; CAD model at a wrong pose; its silhouette projected with the crop camera
(orange) and the residual against the visible mask (pink). The pose moves along the geodesic to the GT; the residual area
curve falls (real raster values). Refiners named; three requirements; the question: compare inside every step, in feature
space?

**S06 The tokens we compare in (~59 s).** 224x224 crop, visible mask; 16x16 patch grid; patches fly into a grid and pool
2x2 into the real 8x8 token grid (PCA colours); one token's 384 real values as a colour strip; weight grid (real w) and
w_p = w_0 + m_p; crop camera: camera centre, image plane, the crop on it, rays through its corners (real intrinsics).

**S07 HERO: the feature field and its energy (~180 s).** F_phi(g, k, a) -> (z_hat, o, h). Object box (r_o) with the CAD
model, which fades (only r_o used); the three real learned tri-planes on the box walls; camera orbit. A point x with its
projections onto the planes; f(x) = sum bilinear, (sigma, h) = MLP([f, gamma(x)]). 256 real rays through the box at the GT
pose with 32 samples each. One real ray: bar chart of sigma_j, transmittance T_j curve, weights w_j (real values), sample
dots sized by weight; o = sum w, h_bar = sum w h / o; real 16x16 opacity map. Appearance code from mean/std of the observed
tokens; f_hat = o * head((1+s(a)) h_bar + b(a), ...) + (1-o) bg(a) (as in the code). Three 8x8 grids: observed, rendered at
GT, rendered 40 deg off; energy equation; weighted per-token squared differences and E bars. The real energy landscape as a
3-D surface: two valleys, at the GT and at the 180 deg copy; a marker moves along the slice: "exact in the pose".

**S08 Fitting the field (~35 s).** Real viewing directions of the training crops around the box; loss min E(g*; z).
Learned density (top 12% of a 40^3 grid, coloured by h) emerges; CAD model faded in for comparison and out again. "phi:
fitted once (100k steps), then frozen".

**S09 Comparing inside every step (~78 s).** z and the field's rendering at g_t. One input token: five real strips (LN z_p,
LN z_hat_p, LN difference, w_p, LN h_p; 1281 dims) -> Lin + e_p. Conditioning token c and token rows through 4 adaLN
blocks; output at c; stop-gradient. Real run: LaRC sample 0 moving in the ball, its rendering and the per-token difference
updating every step (difference shrinks). LaRC vs control side by side from the same priors and noise (real runs,
captioned as an example).

**S10 Selecting among samples (~29 s).** 16 real final LaRC samples (CAD vertices projected onto the crop), energy bars,
sorted by E; lowest E returned; the sample that went astray has the highest E (this run only, captioned); consensus rule.

**S11 Flow instead of diffusion, in progress (~57 s).** Badge "in progress: no results". g_tau = g_0 Exp(tau xi),
xi = Log(g_0^-1 g_1); geodesics from the 16 prior poses of the run to the GT (rotation part); constant body-frame velocity
arrows; loss ||v_theta - xi||^2; Euler update. Left: real 100-step diffusion run; right: 10-step geodesic Euler points
(interpolants, not a trained model, captioned).

**S12 Results (~86 s).** T-LESS 100k x 32 bars (control vs LaRC: VSD, MSSD, MSPD, AR). Translation < 2 cm and rotation
< 5 deg dumbbells. Number line: LiePose published (dagger), LaRC 100k x 32, LiePose our training 400k x 64; detector-crop
GDRNPP behind a dashed separator ("different protocol, not comparable"). SPEED+ log axis: control, LaRC, LiePose (ours),
SPNv3-B (published); lightbox dots, sunlamp squares. Claim: large relative gains; not state of the art.

**S13 What the field can and cannot see (~49 s).** Top view of the real landscape with rings at 3/10/30 deg (sizes only).
P(E(g) > E(g*)) per bin: SPEED+ then T-LESS (paper numbers), chance line, "below 1 deg 0.52". Pick error median 3.4,
mean 14.2. Symmetric tie marked on the landscape.

**S14 Synthesis (~44 s).** The landscape again (slowly turning); energies of the 16 real LaRC samples along the run (15 fall
to the valley floor, one stays high); recap; limits; title.
