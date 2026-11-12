# Narration script: Latent Render-and-Compare (LaRC)

Spoken text, one paragraph per beat. `[Sxx.n]` = scene xx, beat n. Scene code waits for each beat's start, so the voice
and the picture stay in step. The TTS reads this file after the substitutions in `tools/tts.py` (LaRC -> "Lark",
DINOv2 -> "dino v two", ...). Numbers are written as words for the voice; every on-screen number and its source line is in
`out/REPORT.md`.

## S01 The pose problem

[S01.1] Here is a real test image from the T-LESS dataset: a crop of a texture-less industrial part. The task is to recover its six-dimensional pose, the rotation and the translation of the object relative to the camera, from this one crop.

[S01.2] But this part has a symmetry. Turn it by one hundred and eighty degrees about this axis, and the visible shape is the same. So the image does not pin down one pose. It pins down a set of poses: here two, and for a smooth or occluded object, sometimes a whole continuum.

[S01.3] So the answer should be a distribution over poses, not a single value. And the space it lives on is not a flat vector space. It is a curved group. Let's build that space first.

## S02 Poses as a group

[S02.1] A pose is a pair: a rotation R and a translation t. Composing two poses multiplies their four by four transforms, and the set of all of them is the Lie group S E three.

[S02.2] Following LiePose, the pose is expressed relative to the crop. The rotation is allocentric: it removes the apparent rotation caused by where the object sits in the image. The translation is scale invariant: the offset of the projected centre from the crop centre, divided by the box size, and the depth divided by the crop's zoom ratio.

[S02.3] To move on the group, we use its tangent space at the identity, the Lie algebra: six numbers, three for translation and three for rotation. The exponential map wraps a tangent vector onto the group. For the rotation part, a vector omega becomes a turn about the axis of omega, by an angle equal to its length.

[S02.4] The logarithm goes back. Any pose near g can be written as g times the exponential of a small six-vector, and that is exactly how the noise will be added.

## S03 Diffusion on SE(3)

[S03.1] LiePose builds a diffusion process on this group. Take the clean pose g zero. Draw a six-dimensional standard Gaussian, epsilon, scale it by sigma t, and multiply: g t equals g zero times the exponential of sigma t epsilon.

[S03.2] The noise level follows a variance-exploding power law over one hundred steps. This is the actual schedule: sigma rises from one ten-thousandth to one.

[S03.3] Watch what this does to a cloud of poses. Each rotation is a point in a ball: its direction is the rotation axis, and its distance from the centre is the angle, up to pi. As sigma grows, the cloud spreads, until it covers the whole space of rotations.

[S03.4] A network, epsilon theta, is trained to predict that noise from the noisy pose, the step, and the image. Sampling starts from uniformly random poses and applies one hundred updates on the group. Each update estimates the clean pose and steps part of the way toward it.

[S03.5] These are real reverse trajectories: sixteen samples for this crop, from one of our trained denoisers. They start all over the space and contract onto the two symmetric answers.

## S04 What the denoiser never checks

[S04.1] Now look at what that network receives. In LiePose, epsilon theta is a ResNet image encoder followed by Fourier-feature M L P blocks. The image enters once, as a feature vector. The hypothesis g t enters as another vector.

[S04.2] Take the same image and two hypotheses, one close to the truth and one far away. For the network, the two inputs differ only in those six numbers. Whether the object would actually look like this image at g t is never computed. The network has to learn that implicitly, for every pose in S E three.

[S04.3] Every denoising step is open-loop.

## S05 Render-and-compare

[S05.1] The classical way to close the loop is render-and-compare. Take the object's 3D model, render it at the hypothesis, and compare the rendering with the image.

[S05.2] Where the hypothesis is wrong, the residual lights up, and its structure says how to correct the pose. Refiners such as DeepIM, CosyPose, MegaPose and FoundationPose have made this the most accurate final stage on the BOP benchmark for years.

[S05.3] But it needs a renderer and a textured model, it runs as a separate stage after an initial estimate, and on its own it does not represent a multi-modal distribution. LaRC asks whether the comparison can move inside the diffusion, into every step, in a learned feature space.

## S06 The tokens we compare in

[S06.1] First, the space to compare in. The input is a crop of two hundred and twenty-four pixels. On T-LESS, pixels outside the visible mask are set to zero, as in LiePose's protocol.

[S06.2] A frozen DINOv2 vision transformer cuts the crop into fourteen-pixel patches, a sixteen by sixteen grid, and maps each patch to a token with three hundred and eighty-four dimensions. Pooling two by two gives an eight by eight grid: z, sixty-four tokens. The colours show the first three principal components of the real tokens.

[S06.3] Each token gets a weight. w p is w zero plus m p, where m p is the visible mask, max-pooled to the token grid and dilated by one token, and w zero, point zero five, keeps a little weight on the background.

[S06.4] The crop also comes with its crop camera, k: the zoom, the crop origin, the box size and the intrinsics. It holds no pose information, but it lets us cast rays through exactly this crop.

## S07 The feature field and its energy

[S07.1] Now the world model. It must answer one question, for any pose g: which tokens would this object produce, at this pose, in this crop? LaRC answers with a feature field.

[S07.2] Around the object sits a box of half-size r o, one point two times the object's radius. That one number is all that is taken from the CAD model. Inside the box live three axis-aligned planes of learned features, each one hundred and twenty-eight by one hundred and twenty-eight, with thirty-two channels.

[S07.3] For a point x, project it onto each plane, sample bilinearly, and sum. With Fourier features of the position, a two-layer M L P of width one hundred and twenty-eight turns this into a density, sigma of x, and a latent vector, h of x. Outside the box, the density is zero.

[S07.4] To render a hypothesis g, place the box at g in front of the crop camera, and cast one ray through each of sixteen by sixteen sub-patch centres. Along each ray, take thirty-two samples within r o of the object's depth.

[S07.5] This is one real ray of the trained field, at the true pose. The density is near zero in empty space and rises where the ray enters the object. Volume rendering turns it into weights. Alpha j is one minus e to the minus sigma j delta. The transmittance T j is the product of one minus alpha over the samples in front. And w j is T j times alpha j.

[S07.6] The weights sum to the ray's opacity, o, and their weighted average of the latents is the latent of the surface the ray hits. A small head maps that latent and the view direction to a token feature, and rays that miss see a background feature.

[S07.7] Both are modulated by an appearance code, computed from the observation's own token statistics: their mean and standard deviation per channel. The code absorbs lighting and background. It is the same for every hypothesis, so on its own it cannot rank poses.

[S07.8] Pool two by two, and the field outputs z hat of g, the opacity, and the latents, on the same eight by eight grid as the observation. Left, the real observed tokens. Middle, the field's rendering at the true pose. Right, at a pose forty degrees off.

[S07.9] Now compare. The energy is the foreground-weighted squared distance between the rendered and the observed tokens, averaged over channels: E of g, given z.

[S07.10] Sweep the rotation over a slice of rotation space: about the object's symmetry axis along this direction, about a perpendicular axis along that one. This is the real energy of the trained field for this crop. It has two valleys: one at the true pose, and one exactly one hundred and eighty degrees away, at the symmetric copy that the image cannot tell apart.

[S07.11] Because the field renders the pose itself, the rendering is exact in the pose. Turn the hypothesis by one degree, and every sample point moves by one degree in the object frame. Nothing in z hat comes from another image.

## S08 Fitting the field

[S08.1] How is the field learned? Only from posed training crops, one hundred and eighty thousand of them for T-LESS. At each crop's ground-truth pose, the loss is just this energy: the reconstruction error of the tokens.

[S08.2] There is no contrastive term, and the field never sees a wrong pose while fitting. Its density is never supervised. Yet here is the density the trained field has learned inside the box, next to the CAD model it never saw. Its shape is whatever best explains the tokens across all training poses.

[S08.3] The field is fitted once, for one hundred thousand steps, and then frozen.

## S09 Comparing inside every step

[S09.1] Now put the world model into the loop. At step t, for every sample, the frozen field renders at that sample's current noisy pose, g t.

[S09.2] For each patch p, the denoiser builds one input token: the layer-normalised observation, the rendering, their difference, the weight, and the field's latent, concatenated and projected, plus a position embedding. The latent tells the denoiser which part of the object the hypothesis places under that patch.

[S09.3] A conditioning token carries Fourier embeddings of the step and of the noisy pose. It is prepended, and it modulates four adaptive layer-norm transformer blocks of width two hundred and fifty-six. The output at that token is the six-dimensional noise. The rendering is a constant: no gradient flows into the field.

[S09.4] Here is a real run on our crop. The rendering at the current hypothesis, and the difference grid. As the pose converges, the difference shrinks. The denoiser can read the error of its own hypothesis, instead of inferring it from image features alone.

[S09.5] The control is the same network without the rendered quantities. It sees only the observation and the weights, with the same conditioning, budget and diffusion process, so any difference is due to the comparison alone. Here both start from the same prior samples, with the same noise.

## S10 Selecting among samples

[S10.1] A sampler returns one pose per run, and independent runs can land in different modes, such as a flipped configuration. So LaRC draws sixteen samples from independent priors.

[S10.2] The same energy ranks them, using only the observation and the field: return the sample with the lowest E. A second rule needs no world model: consensus, the sample with the most others within ten degrees. In this one run, the sample that went astray also has the highest energy.

## S11 Flow instead of diffusion: in progress

[S11.1] One variant is in progress, and it has no results yet: flow matching instead of diffusion. Join the data pose g zero and a prior pose g one by the geodesic on the group: g tau equals g zero times the exponential of tau xi, where xi is the logarithm of g zero inverse times g one.

[S11.2] Along this path the body-frame velocity is constant: it is xi itself. So the network regresses xi with a squared loss. Sampling integrates the flow on the group with Euler steps, from tau equals one back to zero: g becomes g times the exponential of minus delta tau times v theta.

[S11.3] Diffusion paths are noisy and need one hundred steps. Flow paths are geodesics, and ten to twenty steps are planned. The world model is evaluated once per step and per sample, so fewer steps cut the cost directly. And the velocity has a render-and-compare reading: which way to move the pose so that the rendering matches.

## S12 Results

[S12.1] Now the numbers in the paper. On T-LESS, with ground-truth crops and the BOP average recall, at one hundred thousand steps of batch thirty-two, the control reaches forty point five, and LaRC seventy-two point four. All three recalls rise.

[S12.2] The largest gain is in translation: recall within two centimetres rises from twenty-six point eight to eighty point three. Rotation recall within five degrees rises from forty-five point seven to seventy-one point six.

[S12.3] Against LiePose: its published ground-truth-crop recalls average seventy point one. LiePose trained by us at the full budget, with eight times the training images, reaches seventy-three point seven, one point three above LaRC. Methods with detector crops, such as GDRNPP at seventy-eight point six, follow a different protocol, so they are not a head-to-head comparison.

[S12.4] On the SPEED+ spacecraft benchmark, lower is better. At the matched budget, the world model lowers the score from zero point eight five four to zero point six seven two on lightbox, and from zero point seven nine three to zero point six six four on sunlamp. But LiePose trained by us reaches zero point two three four, and published methods go much lower, for example zero point zero five one for SPNv3. On a log axis, the gap is plain.

[S12.5] So the claim is relative. The comparison gives large gains over the identical denoiser without it. It does not set a new state of the art.

## S13 What the world model can and cannot see

[S13.1] What can the field discriminate on its own? Perturb the true pose, and ask how often the energy of the perturbed pose is higher. On SPEED+, the field prefers the truth over perturbations of three to ten degrees in eighty-four percent of cases, and of ten to thirty degrees in ninety-eight. Below one degree, it is at chance.

[S13.2] On T-LESS it is weaker: zero point five three at one to three degrees, zero point six six at three to ten, and zero point eight six at ten to thirty. The pose its energy picks among candidates has a median error of three point four degrees, but a mean of fourteen point two: it makes gross errors.

[S13.3] And for a symmetric object, the true pose and its symmetric copy produce the same image. No energy computed from the image can prefer one, and the BOP metrics count both as correct.

## S14 Synthesis

[S14.1] Back to the landscape. A diffusion model on S E three samples poses, but each of its steps used to be open-loop.

[S14.2] LaRC renders, at every step and for every sample, what the object would look like in DINOv2 space at the current hypothesis, with a feature field fitted once from posed crops: no mesh, no texture, no reference image. The denoiser compares, and the same rendering gives an energy that ranks the samples.

[S14.3] Against an identical denoiser without the comparison, the gains are large. The method relies on ground-truth crops, it is slower than LiePose because it renders at every step, and its precision is bounded by what pooled DINOv2 tokens can tell apart. Those limits are where the next work lies.
