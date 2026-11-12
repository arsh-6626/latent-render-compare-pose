from common import *


def cells(vals, color, w=0.32, h=0.32, vmax=2.0):
  g = VGroup()
  for i, v in enumerate(vals):
    a = float(np.clip(abs(v) / vmax, 0.12, 1))
    g.add(Square(w).set_fill(color, a).set_stroke(GREY_B, 1).move_to(DOWN * i * (h + 0.04)))
  return g.center()


class S04(LScene):
  SCENE = "S04"

  def construct(self):
    tr, tk, ry = load("traj.npz"), load("tokens.npz"), load("rays.npz")
    rt = tr["field_nowm_rt"]  # real noisy hypotheses along the control run (Lie-algebra coordinates, 6-d)
    rng = np.random.default_rng(3)

    # ---- beat 1: what eps_theta receives (LiePose: ResNet-34 + Fourier-feature MLP)
    d = self.beat(1)
    crop = ImageMobject(os.path.join(DATA, "crop.png"), height=2.6).move_to(LEFT * 5.2 + UP * 1.2)
    il = T("I", 40, OBS_C).next_to(crop, UP, 0.15)
    self.play(FadeIn(crop), FadeIn(il), run_time=1.0)
    feat = cells(rng.normal(size=12), OBS_C, 0.26, 0.26).move_to(LEFT * 2.3 + UP * 1.2)
    fl = TT(r"ResNet-34 features (schematic)", 24, OBS_C).next_to(feat, UP, 0.2)
    cc = crop.copy()
    self.play(cc.animate.scale(0.1).move_to(feat).set_opacity(0), FadeIn(feat, lag_ratio=0.1), FadeIn(fl), run_time=1.6)
    self.remove(cc)
    gv = cells(rt[60, 0], POSE_C).move_to(LEFT * 2.3 + DOWN * 2.0)
    gl = T(r"g_t\in\mathbb{R}^6", 34, POSE_C).next_to(gv, LEFT, 0.3)
    self.play(FadeIn(gv, LEFT * 0.4), FadeIn(gl), run_time=1.2)
    # a small MLP as layers of units; activations flow left to right
    layers = [6, 9, 9, 6]
    xs = [0.6, 2.0, 3.4, 4.8]
    units = [VGroup(*[Circle(radius=0.13).set_stroke(GREY_A, 1.5).move_to(np.array([x, (n - 1) / 2 * 0.42 - i * 0.42, 0]))
                      for i in range(n)]) for x, n in zip(xs, layers)]
    edges = VGroup(*[Line(a.get_center(), b.get_center()).set_stroke(GREY_D, 0.8, 0.6)
                     for l1, l2 in zip(units[:-1], units[1:]) for a in l1 for b in l2])
    net = VGroup(edges, *units).move_to(RIGHT * 2.0 + DOWN * 0.4)
    nl = TT(r"Fourier-feature MLP blocks", 24, GREY_A).next_to(net, UP, 0.25)
    out = cells(rng.normal(size=6) * 0.8, NOISE_C).next_to(net, RIGHT, 0.6)
    ol = T(r"\hat\epsilon", 40, NOISE_C).next_to(out, UP, 0.2)
    a1 = Arrow(feat.get_right(), net.get_left() + UP * 0.6, buff=0.15).set_color(OBS_C)
    a2 = Arrow(gv.get_right(), net.get_left() + DOWN * 0.6, buff=0.15).set_color(POSE_C)
    self.play(ShowCreation(net), FadeIn(nl), GrowArrow(a1), GrowArrow(a2), run_time=1.6)
    for l in units:
      self.play(l.animate.set_fill(YELLOW, 0.8), run_time=0.25)
      self.play(l.animate.set_fill(BG, 0), run_time=0.15)
    self.play(FadeIn(out, RIGHT * 0.3), FadeIn(ol), run_time=0.8)
    eq = T(r"\hat\epsilon=\epsilon_\theta(g_t,\ t,\ I)", 44).to_edge(DOWN, buff=0.45)
    self.play(Write(eq), run_time=1.2)
    self.until(1.0)

    # ---- beat 2: two hypotheses: only the six numbers differ
    d = self.beat(2)
    self.play(FadeOut(eq), run_time=0.5)
    near, far = rt[100, 0], rt[3, 5]  # a converged and an early (far) hypothesis of the real run
    gv2 = cells(far, POSE_C).move_to(gv)
    mA = mesh5(scale=11).apply_matrix(CV2M @ ry["gt_R_cam"], about_point=ORIGIN).move_to(LEFT * 4.6 + DOWN * 2.0)
    Rf = ry["gt_R_cam"] @ tk["R_gt"].T @ lm.se3_exp(far)[0]  # allocentric -> egocentric with the GT translation's correction
    mB = mesh5(scale=11).apply_matrix(CV2M @ Rf, about_point=ORIGIN).move_to(LEFT * 4.6 + DOWN * 2.0)
    lA = TT("near the truth", 24, GT_C).next_to(mA, DOWN, 0.15)
    lB = TT("far away", 24, NOISE_C).next_to(mB, DOWN, 0.15)
    self.play(FadeOut(gl), FadeIn(mA), FadeIn(lA), Transform(gv, cells(near, POSE_C).move_to(gv)), run_time=1.2)
    self.wait(0.6)
    same = TT(r"identical", 26, OBS_C).next_to(feat, RIGHT, 0.25)
    self.play(FadeOut(mA), FadeOut(lA), FadeIn(mB), FadeIn(lB), Transform(gv, gv2), Indicate(gv, color=WHITE), run_time=1.4)
    self.play(FadeIn(same), Indicate(feat, scale_factor=1.05, color=OBS_C), run_time=1.0)
    q = TT(r"would the object look like $I$ at $g_t$?", 34, TXT_C).move_to(RIGHT * 2.6 + DOWN * 2.7)
    nc = TT(r"never computed", 34, NOISE_C).next_to(q, DOWN, 0.15)
    self.until(0.45)
    self.play(FadeIn(q, UP * 0.2), run_time=1.0)
    self.play(FadeIn(nc), run_time=0.6)
    self.until(1.0)

    # ---- beat 3: open loop
    d = self.beat(3)
    self.play(Indicate(nc, scale_factor=1.15), run_time=min(1.2, d))
    self.until(1.0)
    self.wait(0.6)
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
