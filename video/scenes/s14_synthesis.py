from common import *


class S14(L3Scene):
  SCENE = "S14"

  def construct(self):
    tr = load("traj.npz")
    Et = tr["field_full_lat_E_traj"]  # (21, 16): energy of each LaRC sample every 5 steps of the real run
    frame = self.frame

    # ---- beat 1: back to the landscape
    d = self.beat(1)
    surf, f = landscape_surface()
    gts = Group(*[Sphere(radius=0.12).set_color(GT_C).move_to(f(a, 0) + OUT * 0.15) for a in (-180, 0, 180)])
    land = Group(surf, gts).scale(0.8, about_point=ORIGIN)
    frame.reorient(0, 52, 0)
    frame.set_height(9).move_to(RIGHT * 2.6 + OUT * 1.2 + DOWN * 0.6)
    self.play(ShowCreation(surf), run_time=0.6 * d)
    land.add_updater(lambda m, dt: m.rotate(0.12 * dt, axis=OUT, about_point=ORIGIN))
    self.play(FadeIn(gts), run_time=0.8)
    self.until(1.0)

    # ---- beat 2: what LaRC does, and the energy along the real sampling run
    d = self.beat(2)
    ax = small_axes([0, 100, 25], [1.2, 5.2, 1.0], 5.0, 2.9, [2.2, 3.2, 4.2, 5.2], [0, 25, 50, 75, 100], 20).to_corner(UR, buff=0.45)
    axl = TT(r"$E$ of each of the 16 LaRC samples along the run (real)", 22, REN_C).next_to(ax, DOWN, 0.15)
    stp = TT("step", 20, GREY_B).next_to(ax, DOWN, 0.02).align_to(ax, RIGHT)
    fix(ax, axl, stp)
    self.play(FadeIn(ax), FadeIn(axl), FadeIn(stp), run_time=0.8)
    xs = np.arange(0, 101, 5)
    curves = VGroup(*[VMobject().set_points_as_corners([ax.c2p(x, float(np.clip(e, 1.2, 5.2))) for x, e in zip(xs, Et[:, j])])
                      .set_stroke(REN_C, 2, 0.8) for j in range(Et.shape[1])])
    fix(curves)
    self.play(ShowCreation(curves, lag_ratio=0.02), run_time=0.3 * d)
    rec = VGroup(TT(r"render $\hat{\mathbf z}(g_t)$ with a feature field fitted once from posed crops", 28),
                 TT(r"no mesh, no texture, no reference image", 28, GREY_A),
                 TT(r"compare with $\mathbf z$ inside every step; rank samples by $E(g;\mathbf z)$", 28))
    rec.arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_corner(DL, buff=0.45)
    fix(rec)
    for r in rec:
      self.play(FadeIn(r, UP * 0.15), run_time=0.8)
    self.until(1.0)

    # ---- beat 3: claim and limits
    d = self.beat(3)
    self.play(FadeOut(rec), run_time=0.6)
    lim = VGroup(TT(r"large gains over the identical denoiser without the comparison", 30, REN_C),
                 TT(r"limits: ground-truth crops $\cdot$ slower than LiePose (renders every step) $\cdot$", 28, GREY_A),
                 TT(r"precision bounded by what pooled DINOv2 tokens can tell apart", 28, GREY_A))
    lim.arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_corner(DL, buff=0.45)
    fix(lim)
    for r in lim:
      self.play(FadeIn(r, UP * 0.15), run_time=0.9)
    self.until(0.85)
    self.play(FadeOut(Group(ax, axl, stp, curves)), run_time=0.6)
    title = TT(r"Latent Render-and-Compare: SE(3) Diffusion with a Feature-Field World Model", 32).to_edge(UP, buff=0.3)
    fix(title)
    self.play(FadeIn(title), run_time=1.0)
    self.until(1.0)
    self.wait(1.2)
    land.clear_updaters()
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not frame])), run_time=1.2)
    self.finish(tail=0.5)
