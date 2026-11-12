from common import *


class S05(LScene):
  SCENE = "S05"

  def construct(self):
    ry = load("rays.npz")
    Rw, Rg = ry["wr_R_cam"], ry["gt_R_cam"]
    w = lm.so3_log(Rw.T @ Rg)
    res = np.load(os.path.join(DATA, "rc", "residual_frac.npy"))
    N = len(res)

    # ---- beat 1: render the model at the hypothesis, compare with the image
    d = self.beat(1)
    cc = LEFT * 3.2 + DOWN * 0.2
    crop = ImageMobject(os.path.join(DATA, "crop.png"), height=5.4).move_to(cc)
    box = SurroundingRectangle(crop, buff=0).set_stroke(GREY_B, 1.5)
    il = TT("image", 28, OBS_C).next_to(box, UP, 0.12)
    self.play(FadeIn(crop), ShowCreation(box), FadeIn(il), run_time=1.0)
    mc = RIGHT * 3.8 + UP * 0.6
    mesh = mesh5(scale=26).apply_matrix(CV2M @ Rw, about_point=ORIGIN).move_to(mc)
    ml = TT(r"3D model at the hypothesis $g$", 28, REN_C).next_to(mesh, UP, 0.45)
    self.play(FadeIn(mesh), FadeIn(ml), run_time=1.2)
    ovs = Group(*[ImageMobject(os.path.join(DATA, "rc", f"rc_{i:02d}.png"), height=5.4).move_to(cc) for i in range(N)])
    for o in ovs:
      o.set_opacity(0)
    self.add(ovs)
    a_r = Arrow(mesh.get_left(), box.get_right() + UP * 1.2, buff=0.2, path_arc=0.4).set_color(REN_C)
    rl = TT("render", 28, REN_C).next_to(a_r, UP, 0.05)
    self.play(GrowArrow(a_r), FadeIn(rl), ovs[0].animate.set_opacity(1), run_time=1.5)
    leg = VGroup(VGroup(Square(0.25).set_fill(REN_C, 0.5).set_stroke(width=0), TT("rendered silhouette", 24)).arrange(RIGHT, buff=0.15),
                 VGroup(Square(0.25).set_fill(DIFF_C, 0.95).set_stroke(width=0), TT(r"residual $\ne$ image", 24)).arrange(RIGHT, buff=0.15))
    leg.arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(box, DOWN, 0.2).align_to(box, LEFT)
    self.play(FadeIn(leg), run_time=0.8)
    self.until(1.0)

    # ---- beat 2: update the pose until the residual vanishes (geodesic from the wrong pose to the truth)
    d = self.beat(2)
    ax = small_axes([0, 1, 0.5], [0, 0.08, 0.04], 3.6, 2.0, [0.04, 0.08], [0, 1], 20).move_to(RIGHT * 3.8 + DOWN * 2.3)
    axl = TT("residual area", 22, DIFF_C).next_to(ax, UP, 0.1)
    self.play(FadeIn(ax), FadeIn(axl), run_time=0.8)
    al = ValueTracker(0)
    prev = [0.0]

    def mesh_upd(m):
      a = al.get_value()
      Rn, Ro = Rw @ lm.so3_exp(a * w), Rw @ lm.so3_exp(prev[0] * w)
      m.apply_matrix(CV2M @ Rn @ Ro.T @ CV2M.T, about_point=mc)
      prev[0] = a

    mesh.add_updater(mesh_upd)

    def ov_upd(g):
      i = int(round(al.get_value() * (N - 1)))
      for j, o in enumerate(g):
        o.set_opacity(1 if j == i else 0)

    ovs.add_updater(ov_upd)
    graph = VMobject().set_stroke(DIFF_C, 3)
    graph.add_updater(lambda m: m.set_points_as_corners(
      [ax.c2p(x, y) for x, y in zip(np.linspace(0, 1, N), res)][:max(2, int(round(al.get_value() * (N - 1))) + 1)]))
    self.add(graph)
    self.play(al.animate.set_value(1), run_time=0.55 * d, rate_func=smooth)
    mesh.clear_updaters()
    ovs.clear_updaters()
    graph.clear_updaters()
    refs = TT(r"DeepIM \quad CosyPose \quad MegaPose \quad FoundationPose", 28, GREY_A).to_edge(UP, buff=0.3)
    self.play(FadeIn(refs, DOWN * 0.2), run_time=1.0)
    self.until(1.0)

    # ---- beat 3: what it needs, and the question
    d = self.beat(3)
    lims = VGroup(TT(r"$\cdot$ a renderer and a textured 3D model", 30),
                  TT(r"$\cdot$ a separate stage after an initial estimate", 30),
                  TT(r"$\cdot$ one hypothesis, not a distribution", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    lims.move_to(RIGHT * 3.4 + UP * 0.4)
    self.play(FadeOut(Group(a_r, rl, refs, ax, axl, graph)), mesh.animate.scale(0.55).move_to(RIGHT * 5.6 + DOWN * 2.4), run_time=1.0)
    for l in lims:
      self.play(FadeIn(l, RIGHT * 0.3), run_time=0.7)
      self.wait(0.15 * d / 3)
    self.until(0.62)
    qq = TT(r"compare inside \emph{every} denoising step,\\ in a learned feature space?", 36, REN_C).to_edge(DOWN, buff=0.35)
    qq.shift(RIGHT * 2.6)
    self.play(Write(qq), run_time=2.0)
    self.until(1.0)
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
