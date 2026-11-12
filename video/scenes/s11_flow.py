from common import *


class S11(L3Scene):
  SCENE = "S11"

  def construct(self):
    tk, tr = load("tokens.npz"), load("traj.npz")
    R_gt, t_gt = tk["R_gt"], tk["t_gt"].astype(np.float64)
    rt0 = tr["field_full_lat_rt"][0]  # the 16 prior poses of the real run (uniform rotation, translation in [-1, 1]^3)
    g0 = (R_gt.astype(np.float64), t_gt)
    frame = self.frame

    # ---- beat 1: the geodesic interpolant on SE(3)
    d = self.beat(1)
    badge = TT(r"in progress: a variant with no results yet", 30, FLOW_C).to_corner(UR, buff=0.35)
    badge_box = SurroundingRectangle(badge, buff=0.12).set_stroke(FLOW_C, 2)
    fix(badge, badge_box)
    eq = T(r"g_\tau = g_0\,\mathrm{Exp}(\tau\,\xi),\qquad \xi=\mathrm{Log}\big(g_0^{-1}g_1\big),\quad \tau\in[0,1]", 42,
           t2c={"g_0": GT_C, r"\xi": FLOW_C, "g_1": POSE_C}).to_edge(UP, buff=0.9)
    fix(eq)
    self.play(FadeIn(badge), ShowCreation(badge_box), run_time=1.0)
    self.play(Write(eq), run_time=2.0)
    bc = ORIGIN
    bl = ball(radius=2.4).move_to(bc)
    sc = 2.4 / BALL_R
    paths = []
    xis = []
    for j in range(16):
      g1 = lm.se3_exp(rt0[j])
      taus = np.linspace(1, 0, 61)
      pts = [bc + rot_to_ball(lm.geodesic(g0, g1, ta)[0][0]) * sc for ta in taus]
      paths.append(np.array(pts))
      xis.append(lm.geodesic(g0, g1, 0.0)[1])
    gtd = Sphere(radius=0.09).set_color(GT_C).move_to(bc + rot_to_ball(R_gt) * sc)
    frame.reorient(-20, -10, 0)
    frame.shift(UP * 0.7)
    lab = TT(r"rotation part of $g_\tau$ for 16 prior poses $g_1$ (the paths the network is trained on)", 26, GREY_A).to_edge(DOWN, buff=0.3)
    fix(lab)
    pg = PathGrower(paths, FLOW_C, 2.2, 0.9, head_r=0.07, head_color=POSE_C)
    self.play(FadeIn(bl), FadeIn(gtd), FadeIn(pg.heads), FadeIn(lab), run_time=1.2)
    self.add(pg.lines)
    self.play(pg.prog.animate.set_value(60), run_time=0.4 * d, rate_func=smooth)
    pg.stop()
    self.until(1.0)

    # ---- beat 2: constant body-frame velocity; regression and Euler steps on the group
    d = self.beat(2)
    # tangent arrows along one path (direction of motion toward g_0, i.e. -xi)
    j = 3
    p = paths[j]
    arrows = VGroup()
    for k in (8, 20, 32, 44):
      a, b = p[k], p[k + 3]
      arrows.add(Arrow(a, a + 2.2 * (b - a), buff=0, stroke_width=4).set_color(FLOW_C))
    self.play(frame.animate.shift(LEFT * 3.0), run_time=1.0)
    self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2), run_time=1.4)
    e1 = T(r"\frac{d}{d\tau}\,g_\tau = g_\tau\,\xi^\wedge\ \ (\text{constant body-frame velocity})", 34)
    e2 = T(r"\mathcal L=\big\|v_\theta(g_\tau,\tau,I)-\xi\big\|^2", 36, t2c={r"\xi": FLOW_C})
    e3 = T(r"g\ \leftarrow\ g\,\mathrm{Exp}\big(-\Delta\tau\,v_\theta\big),\quad \tau: 1\to 0", 36, t2c={"v_\\theta": FLOW_C})
    col = VGroup(e1, e2, e3).arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_corner(UL, buff=0.4).shift(DOWN * 1.5)
    fix(col)
    self.play(FadeOut(eq), FadeIn(e1, DOWN * 0.2), run_time=1.4)
    self.until(0.4)
    self.play(FadeIn(e2, DOWN * 0.2), run_time=1.2)
    self.until(0.65)
    self.play(FadeIn(e3, DOWN * 0.2), run_time=1.2)
    self.until(1.0)

    # ---- beat 3: noisy 100-step diffusion vs 10 straight Euler steps (planned)
    d = self.beat(3)
    self.play(FadeOut(Group(*[m for m in self.mobjects if m not in (frame, badge, badge_box)])), run_time=0.8)
    frame.reorient(0, 0, 0)
    frame.move_to(ORIGIN)
    r2 = 2.0
    s2 = r2 / BALL_R
    cL, cR = LEFT * 3.4 + DOWN * 0.4, RIGHT * 3.4 + DOWN * 0.4
    bL, bR = ball(radius=r2).move_to(cL), ball(radius=r2).move_to(cR)
    R_l = tr["field_full_lat_R"]
    dpaths = [np.array([cL + rot_to_ball(R_l[i, jj]) * s2 for i in range(101)]) for jj in range(16)]
    fpaths = [np.array([cR + rot_to_ball(lm.geodesic(g0, lm.se3_exp(rt0[jj]), ta)[0][0]) * s2 for ta in np.linspace(1, 0, 11)])
              for jj in range(16)]
    tL = TT(r"diffusion: 100 stochastic steps (real run)", 28, REN_C).next_to(bL, UP, 0.3)
    tR = TT(r"flow: 10--20 Euler steps (planned)", 28, FLOW_C).next_to(bR, UP, 0.3)
    gts = Group(*[Sphere(radius=0.07).set_color(GT_C).move_to(c + rot_to_ball(R_gt) * s2) for c in (cL, cR)])
    gL = PathGrower(dpaths, REN_C, 1.5, 0.7, head_r=0.06, head_color=POSE_C)
    gR = PathGrower(fpaths, FLOW_C, 2.0, 0.9, head_r=0.06, head_color=POSE_C)
    steps = DotCloud(np.concatenate([fp for fp in fpaths]), radius=0.04).set_color(FLOW_C).set_opacity(0.0)
    self.play(FadeIn(bL), FadeIn(bR), FadeIn(tL), FadeIn(tR), FadeIn(gts), FadeIn(gL.heads), FadeIn(gR.heads), run_time=1.0)
    self.add(gL.lines, gR.lines, steps)
    gR.prog.add_updater(lambda m: m.set_value(gL.prog.get_value() / 10))
    self.add(gR.prog)
    self.play(gL.prog.animate.set_value(100), run_time=0.45 * d, rate_func=linear)
    gR.prog.clear_updaters()
    gL.stop()
    gR.stop()
    self.play(steps.animate.set_opacity(1), run_time=0.6)
    cost = TT(r"the world model runs once per step and per sample: fewer steps, lower cost", 26, GREY_A).to_edge(DOWN, buff=0.95)
    cap = caption(r"right: the geodesic interpolants from the same priors (mathematics of the variant), not a trained model")
    fix(cost, cap)
    self.play(FadeIn(cost), FadeIn(cap), run_time=1.0)
    self.until(1.0)
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not frame])), run_time=0.8)
    self.finish()
