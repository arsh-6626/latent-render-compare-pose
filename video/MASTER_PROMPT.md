# MASTER PROMPT: Research Paper Explainer Video in the Style of 3Blue1Brown

## TASK

Create a technically rigorous, visually exceptional, approximately 10-minute mathematical explainer video for the research paper:

**Paper title:** {{PAPER_TITLE}}

**Proposed method/system:** {{METHOD_NAME}}

**Main baseline(s):** {{BASELINE(S)}}

The objective is to produce an immersive, research-grade mathematical explanation inspired by the visual storytelling, mathematical intuition, and animation quality of 3Blue1Brown.

This is NOT a conventional research presentation, narrated slideshow, PowerPoint-style video, or architecture-diagram walkthrough.

It must be an actual mathematical explainer in which the underlying mathematics, geometry, algorithms, transformations, and computational processes come alive through carefully constructed animations.

The audience consists of researchers and graduate-level students in the relevant field. Assume mathematical maturity. Explain the research in considerable technical depth without sacrificing visual clarity.

**The central philosophy: Do not merely tell the audience what the method does. Make them watch the mathematics happen.**

---

# 1. DELIVERABLE AND TECHNICAL REQUIREMENTS

<badge color="success">Mandatory deliverable: One rendered MP4</badge>

Produce ONE complete, standalone video.

- Format: MP4 (H.264).
- Resolution: 1920 × 1080 minimum.
- Frame rate: 60 FPS preferred, 30 FPS minimum.
- Duration: Approximately 10 minutes. Acceptable range: 7–15 minutes.
- Animation framework: 3b1b/manim (https://github.com/3b1b/manim).
- Audio: Continuous, synchronized, high-quality voiceover.
- Aspect ratio: 16:9.
- Output location: `{{OUTPUT_PATH}}`.

### Mandatory constraints

1. All mathematical and geometric animations must be implemented using Manim.
2. Deliver a fully rendered video, not an HTML page, web player, collection of images, or presentation.
3. Do not substitute static diagrams or slides for animations.
4. Do not artificially stretch scenes or slow down existing animations to achieve the target duration. Add genuinely useful explanatory scenes instead.
5. All scenes must be integrated into one coherent video with consistent visual language.
6. The final MP4 must be inspected for rendering, synchronization, visual, and audio problems before delivery.

Supporting source code, scripts, storyboards, and intermediate assets may be retained, but they do not replace the MP4.

---

# 2. RESEARCH UNDERSTANDING AND FIDELITY

Before creating anything, thoroughly analyze the complete research paper.

Do not start by making animations from the abstract or introduction alone.

Extract and understand:

- The research problem and its formal mathematical formulation.
- The motivation and limitations of existing approaches.
- The paper's actual contributions and novelty.
- The central mathematical ideas.
- All important equations, operators, and objective functions.
- The model architecture and computational pipeline.
- The algorithms and implementation details.
- The assumptions and mathematical constraints.
- The experimental setup, datasets, metrics, baselines, and results.
- The qualitative results and failure cases.

### Research fidelity rules

These are hard constraints:

- The paper is the primary source of truth.
- Preserve the original mathematical notation.
- Never invent equations, model components, experimental results, or implementation details.
- Do not confuse conceptual intuition with the actual mathematical formulation.
- Do not silently simplify a mathematical operation in a way that changes its meaning.
- Verify dimensions, coordinate systems, transformations, and mathematical assumptions.
- Distinguish explicitly between the paper's actual implementation and any explanatory interpretation introduced for visualization.
- When the paper omits an implementation detail, do not fabricate it.
- If additional background is needed, use established mathematical definitions and clearly distinguish them from the paper's contributions.

Whenever possible, reconstruct the mathematical operations computationally and verify them independently before animating them.

**The objective is to create a faithful visual explanation of the research, not an aesthetically pleasing reinterpretation of it.**

---

# 3. CRITICAL VISUAL DIRECTIVE: FULL-SCREEN MATHEMATICAL STORYTELLING

<badge color="danger">Highest-priority visual constraint</badge>

The video must be animation-first.

The screen must be filled with large, immersive, evolving mathematical visualizations. The audience should feel that they are exploring the mathematics in a visual environment, not watching a presentation.

## 3.1 Use the entire screen

- Treat the entire 16:9 canvas as a mathematical storytelling space.
- Use large, expansive geometric constructions and diagrams.
- Let mathematical objects grow, move, rotate, transform, deform, project, and interact across the screen.
- Use camera movement and spatial transitions to reveal different aspects of mathematical structures.
- Make the central mathematical visualization occupy most of the available screen.
- Use full-screen scenes instead of restricting animations to small central regions.
- Let diagrams evolve naturally rather than remaining confined to a fixed layout.
- Use depth, perspective, scale, and spatial composition to make complex ideas intuitive.

The animation must be visually dominant throughout the video.

## 3.2 Absolutely prohibit presentation-style layouts

DO NOT create:

- Grids of rectangular cells containing text.
- Collections of rectangular information cards.
- Dashboard-like layouts.
- Architecture diagrams consisting of static boxes and arrows.
- A central diagram surrounded by multiple explanatory text boxes.
- Multiple panels containing paragraphs.
- Full-screen bullet-point slides.
- A tiny animated object surrounded by empty space and text.
- Static text-heavy screens while narration continues.
- Repeated rectangular blocks with different labels pretending to be animations.

**A grid of boxes containing labels is NOT an acceptable substitute for an animated mathematical or computational process.**

Do not use conventional presentation design as the default scene composition.

## 3.3 Animate the actual concepts

Every important concept should be represented by a meaningful visual sequence.

Examples:

- Instead of showing a neural architecture as rectangular blocks, animate the actual flow of representations through its components.
- Instead of displaying a transformation matrix, show the geometric transformation it performs.
- Instead of showing a static manifold, construct it, rotate it, reveal its curvature, and demonstrate the relevant projections.
- Instead of showing an algorithm as a flowchart, animate its intermediate computations.
- Instead of showing an optimization equation, animate the optimization process and its trajectory.
- Instead of displaying a static probability distribution, animate its evolution, sampling, or parameter changes.
- Instead of presenting a comparison table, demonstrate competing methods operating on the same inputs.
- Instead of displaying a static attention matrix, animate the relevant attention relationships and information flow.

These are examples. Use visualizations appropriate to the actual paper.

## 3.4 Text must support the animation

Text and equations are important, particularly for research-level explanations.

However:

- Never allow explanatory paragraphs to dominate the screen.
- Introduce equations progressively.
- Display mathematical notation prominently when it is being explained.
- Use concise annotations, labels, and callouts.
- Integrate equations with the relevant geometric or computational visualization.
- Use spatial positioning to reveal relationships between mathematical objects.
- When a complex derivation requires substantial notation, dedicate the screen to the derivation while progressively constructing and transforming it.

The text must explain the visualization, not replace it.

## 3.5 Continuous visual development

Every scene must have a meaningful visual progression.

A typical scene should involve:

1. Introducing a mathematical object or problem.
2. Constructing its representation.
3. Demonstrating the actual operation.
4. Revealing the resulting behavior.
5. Connecting the visual result to the relevant mathematical equation.
6. Explaining why the result matters.

Avoid introducing an object and leaving it motionless while the narrator talks for an extended period.

Intentional static moments are acceptable when they improve comprehension, such as pausing on a completed derivation or highlighting a geometric relationship. However, static presentation must never become the dominant visual style.

### Final visual acceptance test

For every scene, ask:

"If all the text were hidden, would the animation still communicate a meaningful mathematical or computational idea?"

If not, redesign the scene.

The finished video should feel like an immersive mathematical exploration, not a narrated PowerPoint presentation.

---

# 4. MATHEMATICAL AND GEOMETRIC ACCURACY

This is a research explainer. Visual accuracy is as important as mathematical accuracy.

Every important mathematical visualization must be generated from the actual mathematical definitions and equations.

### Examples of required correctness

**Linear algebra**
- Perform actual matrix operations.
- Represent vectors and transformations correctly.
- Preserve dimensionality and coordinate conventions.
- Visualize basis changes and projections accurately.

**Geometry**
- Use actual geometric constructions.
- Preserve distances, angles, and constraints where required.
- Use correct coordinate systems.
- Represent manifolds and their embeddings faithfully.

**3D transformations**
- Use actual rotation and transformation functions.
- Animate genuine 3D objects.
- Use mathematically correct projections.
- Maintain consistent reference frames.
- Distinguish between coordinate transformations and physical transformations.

**Probability and optimization**
- Generate distributions using their actual parameterizations.
- Represent sampling correctly.
- Animate the specified objective and optimization procedure.
- Show meaningful trajectories rather than arbitrary motion.

**Machine learning**
- Animate actual intermediate representations.
- Show how tensors, embeddings, attention, and latent variables evolve.
- Represent the model's real computational pipeline.
- Distinguish training-time operations from inference-time operations.

Never use arbitrary motion merely to make an equation look interesting.

If a mathematical object is difficult to visualize, invest additional effort in constructing a faithful representation rather than substituting a generic illustration.

---

# 5. NARRATION AND EXPLANATORY STYLE

### Target audience

Researchers, PhD students, graduate students, and technically proficient practitioners in the relevant field.

Assume familiarity with the necessary mathematical background, but explain the paper's specific mathematical constructions carefully.

### Narration philosophy

Use the explanatory style of 3Blue1Brown:

- Intuitive but mathematically rigorous.
- Conversational but intellectually precise.
- Progressive explanations.
- Strong connections between visual intuition and formal mathematics.
- Carefully motivated mathematical derivations.
- Natural transitions between concepts.

Avoid sounding like an academic paper being read aloud.

### Mathematical explanations

For every central equation:

1. Introduce the mathematical objects.
2. Define the variables and their dimensions.
3. Explain the assumptions.
4. Derive the expression step by step where appropriate.
5. Explain the purpose of each mathematical operation.
6. Explain the significance of each term.
7. Demonstrate the operation visually.
8. Connect the result to the overall method.

Do not simply display an equation and read it aloud.

Explain why the equation is formulated that way.

### Voiceover

- Use one consistent, natural-sounding voice.
- Use clear pronunciation of technical terminology.
- Maintain an engaging but measured pace.
- Synchronize narration with the corresponding visual demonstration.
- Avoid long stretches of narration without visual development.
- Avoid excessively rapid explanations of complex mathematics.
- Avoid unnecessary repetition.
- Do not artificially stretch narration to fill the duration.

Write the complete narration script before implementing the animations.

The narration must tell one continuous intellectual story.

---

# 6. VIDEO STRUCTURE AND CONTENT

The following is the recommended narrative structure. Adapt scene boundaries and timing to the paper, but cover every relevant component.

## SCENE 1: The Problem

**Objective: Establish why the research exists.**

Explain:

- What is the underlying problem?
- Why is it important?
- What makes it difficult?
- What are the mathematical or computational obstacles?

Begin with a concrete visual demonstration of the problem.

Use a large, evolving visualization that establishes the central challenge.

Avoid spending excessive time on generic background information.

## SCENE 2: Why Existing Methods Are Insufficient

Introduce the most relevant baseline approaches.

Explain:

- How the existing methods work.
- Their underlying assumptions.
- Their mathematical or computational limitations.
- Where their behavior becomes problematic.

Demonstrate the limitations visually.

The audience should understand the problem with existing approaches before the proposed solution is introduced.

## SCENE 3: The Central Insight

Introduce the key conceptual contribution.

Explain:

- What is the main insight?
- What mathematical structure does the paper exploit?
- What changes compared with existing approaches?
- Why might this formulation work better?

Build the central idea progressively through animation.

Introduce the main mathematical equation only after establishing the intuition behind it.

## SCENE 4: Mathematical Formulation and Derivation

**This section must receive substantial attention.**

Explain the important mathematical foundations in detail.

For every central equation:

- Introduce the variables.
- Explain the dimensions and mathematical spaces involved.
- Explain the assumptions.
- Derive the formulation progressively.
- Explain every important operator.
- Connect the expression to its geometric interpretation.
- Show the actual computation through animation.

Do not skip intermediate mathematical steps merely to save time.

If the method involves manifolds, Lie groups, probability distributions, differential geometry, optimization, latent representations, or other specialized structures, explain the necessary foundations before using them.

The audience should understand not only what the equations say, but why the formulation is mathematically meaningful.

## SCENE 5: How the Method Actually Works

Explain the actual implementation.

Show:

- The input representation.
- The computational pipeline.
- Intermediate representations.
- The transformations performed.
- Interactions between components.
- Training objectives, if relevant.
- Optimization or inference procedures.
- The final output.

Animate the actual flow of information.

Do not use static architecture boxes as a substitute for showing the computation.

Where possible, display the intermediate mathematical states as they are generated.

## SCENE 6: THE HERO VISUALIZATION

**The mathematical centerpiece of the entire video.**

Central mathematical object: {{HERO_VISUALIZATION}}

This should be the most elaborate, visually impressive, and intellectually valuable scene.

Identify the single mathematical or geometric concept that is most important for understanding the proposed method.

Build a dedicated visualization around it.

Requirements:

1. Construct the actual mathematical object.
2. Begin with its simplest representation.
3. Progressively reveal its structure.
4. Introduce relevant coordinate systems.
5. Animate its transformations.
6. Show important geometric relationships.
7. Demonstrate relevant projections or mappings.
8. Connect the geometry to the equations.
9. Show how the method uses this structure computationally.
10. Explain why this construction is important to the contribution.

Use genuine 3D geometry when the mathematical structure is three-dimensional.

Use camera rotations, perspective, zooms, spatial transformations, and multiple geometric viewpoints where appropriate.

Do not make this scene a generic conceptual illustration.

**The goal is for the audience to understand the central mathematical object through visual experience, not merely through verbal explanation.**

## SCENE 7: Method Versus Baselines

Compare the proposed method against:

{{BASELINE(S)}}

Create a controlled visual comparison.

- Use identical initial conditions and inputs wherever applicable.
- Use consistent coordinate systems.
- Preserve comparable camera viewpoints.
- Animate the actual baseline computation.
- Animate the proposed method's computation.
- Highlight the exact mathematical or computational difference.
- Show the resulting behavioral difference.
- Explain the practical significance.

Do not rely solely on tables or numerical claims.

The main advantage should be visible in the animation itself.

## SCENE 8: Experimental Results

Present the headline results:

{{HEADLINE_RESULTS}}

Baseline results:

{{BASELINE_NUMBERS}}

Include relevant:

- Quantitative performance metrics.
- Ablation studies.
- Efficiency improvements.
- Computational cost.
- Training and inference performance.
- Generalization results.
- Statistical qualifications.

Use animated plots, evolving graphs, and direct visual comparisons.

Do not fabricate or round results inaccurately.

Explain what the results establish and what they do not establish.

## SCENE 9: Qualitative Results and Validation

Visualize:

{{QUALITATIVE_RESULTS}}

Show representative examples from the actual paper.

Where applicable:

- Animate the input and output.
- Show intermediate predictions.
- Demonstrate transformations.
- Compare successful and unsuccessful cases.
- Highlight behavior under different conditions.
- Include important failure cases.

Use real experimental examples where available.

Do not manufacture qualitative results.

The objective is to demonstrate the method's actual behavior.

## SCENE 10: Final Synthesis

Return to the original research problem.

Summarize:

- The central difficulty.
- The key mathematical insight.
- The proposed method.
- Why the formulation matters.
- What the experimental results establish.
- The remaining limitations.

Bring back the central hero visualization and connect it to the complete method.

End with a concise, intellectually satisfying explanation of the paper's contribution.

---

# 7. VISUALIZATION PRIORITIES

Depending on the actual paper, prioritize the following types of mathematical animations.

These are possible visualization techniques, not mandatory additions.

### Geometry and 3D mathematics

- Manifolds and embeddings.
- Coordinate systems.
- Geometric projections.
- Rotations and rigid transformations.
- Lie groups and Lie algebras.
- Spherical geometry.
- Differential geometry.
- Geometric constraints.
- Trajectories and transformation spaces.

### Machine learning and representation learning

- Embedding spaces.
- Latent representations.
- Feature transformations.
- Attention mechanisms.
- Information flow.
- Neural network computations.
- Representation alignment.
- Contrastive objectives.
- Predictive representations.

### Optimization and probabilistic modeling

- Loss landscapes.
- Gradient trajectories.
- Probability distributions.
- Sampling procedures.
- Diffusion trajectories.
- Denoising processes.
- Energy functions.
- Optimization constraints.

### Algorithms and computational structures

- Intermediate tensor transformations.
- Data flow.
- Computational graphs.
- Iterative algorithms.
- Search and refinement.
- Forward and backward computations.

Choose only what is relevant to the actual paper.

---

# 8. VISUAL DESIGN LANGUAGE

Maintain a consistent, sophisticated visual identity throughout the video.

### Aesthetic direction

Use a visual language inspired by 3Blue1Brown:

- Dark or otherwise restrained backgrounds.
- High-contrast mathematical objects.
- Carefully selected, consistent colors.
- Clean mathematical typography.
- Smooth transitions.
- Elegant camera movements.
- Purposeful spatial composition.
- Visually distinctive geometric constructions.

Do not blindly copy an existing video's scenes or artwork.

Develop an original visual language suited to the paper.

### Color consistency

Assign consistent colors to mathematical objects.

For example:

- Different coordinate systems should have distinguishable colors.
- Inputs and outputs should have consistent visual identities.
- Different transformations should use distinguishable visual representations.
- Baseline and proposed-method components should be visually distinguishable.

Once a color represents a particular object or concept, preserve that convention throughout the video.

### Typography

- Use properly typeset LaTeX.
- Ensure equations are readable at 1080p.
- Avoid excessive text.
- Introduce notation progressively.
- Use consistent font sizes and mathematical styling.
- Never allow text to overlap important geometry.

---

# 9. PRODUCTION WORKFLOW

Do not attempt to create the entire video as one unstructured animation.

Use the following staged workflow.

## PHASE A: Paper Analysis

Read the paper thoroughly.

Produce an internal research breakdown containing:

- Problem formulation.
- Main contribution.
- Mathematical foundations.
- Important equations.
- Implementation details.
- Experimental results.
- Baselines.
- Central visualization opportunities.

Identify the mathematical ideas that will require custom geometry or computational animation.

## PHASE B: Narrative and Storyboard

Create a complete scene-by-scene storyboard.

For each scene specify:

- Scene objective.
- Narration.
- Mathematical equations.
- Visual composition.
- Animation sequence.
- Mathematical operations to compute.
- Camera movement.
- Transition to the next scene.
- Estimated duration.

Ensure that the narrative flows naturally.

Do not use a generic storyboard template for every scene. Design the visualization around the specific mathematical concept being explained.

## PHASE C: Mathematical Implementation

Before animating complex objects:

- Implement the underlying mathematical functions.
- Verify transformations numerically.
- Verify dimensional consistency.
- Test coordinate conventions.
- Generate geometric objects computationally.
- Compare visualized operations against the paper's equations.

Do not build elaborate animations around unverified mathematical code.

## PHASE D: Manim Implementation

- Implement all scenes using Manim.
- Organize scenes into modular components.
- Reuse mathematical and geometric objects where appropriate.
- Create custom animation functions for complex transformations.
- Use actual computed geometry.
- Synchronize voiceover and animation timing.
- Maintain visual consistency.

Prefer well-designed, substantial scenes over many superficial transitions.

## PHASE E: Scene-Level Rendering

Render scenes individually before assembling the video.

Inspect every scene for:

- Mathematical correctness.
- Geometric accuracy.
- Animation quality.
- Visual clarity.
- Text readability.
- Camera composition.
- Narration synchronization.
- Rendering artifacts.

Rework scenes that fail.

## PHASE F: Final Assembly

- Assemble the scenes into one continuous video.
- Integrate the complete narration.
- Synchronize the audio.
- Normalize voiceover levels.
- Add subtle background music only if appropriate.
- Ensure smooth transitions.
- Verify the final duration.
- Export the final H.264 MP4.

---

# 10. QUALITY ASSURANCE AND ACCEPTANCE CRITERIA

The video is complete only when it satisfies ALL of the following.

### Research quality

- Faithfully represents the actual paper.
- Explains the central contribution correctly.
- Covers the important mathematical foundations.
- Includes the actual implementation details.
- Uses accurate experimental results.

### Mathematical quality

- Equations are correct.
- Variables and dimensions are explained.
- Transformations are mathematically valid.
- Geometric visualizations respect the actual mathematical structures.
- Computational demonstrations implement the stated operations.

### Visual quality

- Full-screen visual storytelling is the dominant presentation style.
- Large, meaningful animations occupy the canvas.
- Important concepts are demonstrated rather than merely described.
- No presentation-style grid of text boxes or information cards.
- No static architecture diagrams replacing actual computations.
- No tiny central animation surrounded by text.
- No extended text-heavy scenes.
- Every major scene has a meaningful visual progression.
- Camera movements and transitions serve an explanatory purpose.

### Explanatory quality

- Narration is clear and technically detailed.
- Mathematical derivations are sufficiently complete.
- Visuals and narration reinforce each other.
- The narrative progresses logically.
- The central contribution is unmistakable.
- The baseline comparison is convincing.

### Technical quality

- Complete MP4 exists at the specified output location.
- Video plays from beginning to end.
- Audio is synchronized.
- No missing scenes.
- No clipping or overlapping elements.
- No rendering artifacts.
- No placeholder content.
- No unfinished transitions.

---

# 11. FINAL INSTRUCTION

Do not optimize for producing the video quickly at the expense of explanatory quality.

Spend the majority of the creative and implementation effort on the mathematical visualizations, particularly the central mathematical contribution.

Avoid superficial animation, static layouts, generic architecture diagrams, and text-heavy presentation scenes.

Every important equation should have a visual interpretation. Every important geometric structure should be constructed faithfully. Every important computational operation should be demonstrated.

The audience should be able to see the mathematics, understand why the method works, and appreciate precisely what the paper contributes.

**The ultimate standard:**

Create a video that a researcher would genuinely enjoy watching because it reveals the mathematics of the paper in a way that is difficult to achieve through reading alone.

It should combine the mathematical depth of a research seminar with the visual clarity and storytelling of a high-quality 3Blue1Brown episode.

**Do not stop at the paper analysis, storyboard, narration script, or partial scenes. Complete the production, render the entire video, inspect it, and deliver the final standalone MP4.**
