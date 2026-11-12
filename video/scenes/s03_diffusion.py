from common import *


class S03(L3Scene):
  SCENE = "S03"

  def construct(self):
    tk, tr = load("tokens.npz"), load("traj.npz")
    R_gt, S = tk["R_gt"], tk["S"]   # allocentric GT rotation of the test crop and the object's symmetry
    sig = lm.sigmas()
    rng = np.random.default_rng(7)
    frame = self.frame

    # ---- beat 1: g_t = g_0 Exp(sigma_t eps)
    d = self.beat(1)
    eq = T(r"g_t \;=\; g_0\,\mathrm{Exp}(\sigma_t\,\epsilon),\qquad \epsilon\sim\mathcal{N}(0,I_6)", 52,
           t2c={"g_0": GT_C, r"\sigma_t": YELLOW, r"\epsilon": NOISE_C, "g_t": POSE_C}).to_edge(UP, buff=0.5)
    fix(eq)
    self.play(Write(eq), run_time=2.0)
    # the noise vector: six real standard-normal draws (3 translation, 3 rotation)
    eps = rng.normal(size=6)
    bars = VGroup()
    for i, e in enumerate(eps):
      b = Rectangle(0.6, abs(e) * 1.3).set_fill(NOISE_C if i >= 3 else TEAL_C, 0.9).set_stroke(width=0)
      b.move_to(LEFT * 5.6 + RIGHT * 0.8 * i + UP * (np.sign(e) * abs(e) * 0.65 - 0.6))
      bars.add(b)
    base = Line(LEFT * 6.0 + DOWN * 0.6, LEFT * 1.2 + DOWN * 0.6).set_stroke(GREY_B, 1.5)
    bl1 = TT(r"$v$ (translation)", 26, TEAL_C).move_to(LEFT * 4.8 + DOWN * 2.9)
    bl2 = TT(r"$\omega$ (rotation)", 26, NOISE_C).move_to(LEFT * 2.4 + DOWN * 2.9)
    eps_l = T(r"\epsilon\in\mathbb{R}^6", 40, NOISE_C).move_to(LEFT * 3.6 + UP * 1.9)
    fix(bars, base, bl1, bl2, eps_l)
    self.play(ShowCreation(base), LaggedStart(*[GrowFromEdge(b, DOWN if b.get_center()[1] > -0.6 else UP) for b in bars], lag_ratio=0.12),
              FadeIn(bl1), FadeIn(bl2), FadeIn(eps_l), run_time=2.0)
    # the object at g_0 and at g_0 Exp(sigma eps) for growing sigma (rotation part as the object turns)
    c0 = RIGHT * 2.8 + DOWN * 0.8
    m0 = mesh5(scale=20).apply_matrix(CV2M @ load("rays.npz")["gt_R_cam"], about_point=ORIGIN).move_to(c0)
    ghost = m0.copy().set_opacity(0.2)
    g0l = T("g_0", 36, GT_C).next_to(m0, DOWN, 0.6)
    fix(g0l)
    self.play(FadeIn(m0), FadeIn(g0l), run_time=1.0)
    self.add(ghost)
    st = ValueTracker(0.0)
    prev = [np.eye(3)]
    Rc = load("rays.npz")["gt_R_cam"]

    def upd(m):
      Rn = Rc @ lm.so3_exp(st.get_value() * eps[3:])  # body-frame noise: R_0 Exp(sigma omega)
      D = CV2M @ Rn @ np.linalg.inv(CV2M @ prev[0])
      m.apply_matrix(D, about_point=c0)
      prev[0] = Rn

    prev[0] = Rc
    m0.add_updater(upd)
    sl = VGroup(T(r"\sigma =", 34, YELLOW), DecimalNumber(0, num_decimal_places=2, font_size=34).set_color(YELLOW)).arrange(RIGHT)
    sl.next_to(m0, UP, 0.9)
    sl[1].add_updater(lambda m: m.set_value(st.get_value()))
    fix(sl)
    self.add(sl)
    self.play(st.animate.set_value(0.25), run_time=0.15 * d)
    self.play(st.animate.set_value(0.7), run_time=0.15 * d)
    self.until(1.0)
    m0.clear_updaters()

    # ---- beat 2: the actual schedule
    d = self.beat(2)
    self.play(FadeOut(Group(bars, base, bl1, bl2, eps_l, m0, ghost, g0l, sl)), run_time=0.8)
    ax = small_axes([0, 100, 25], [0, 1, 0.25], 7.5, 4.2, [0.25, 0.5, 0.75, 1.0], [25, 50, 75, 100]).move_to(DOWN * 0.5)
    xl = T("t", 34).next_to(ax.x_axis, RIGHT, 0.2)
    yl = T(r"\sigma_t", 34, YELLOW).next_to(ax.y_axis, UP, 0.2)
    curve = ax.get_graph(lambda x: float(np.interp(x, np.arange(101), sig)), x_range=[0, 100, 0.5]).set_stroke(YELLOW, 4)
    form = T(r"\sigma_t^2=\Big(\big(1-\tfrac{t}{T}\big)\,a_0^{1/3}+\tfrac{t}{T}\Big)^{3},\ \ a_0=10^{-8},\ T=100", 32).next_to(ax, UP, 0.3)
    self.play(ShowCreation(ax), FadeIn(xl), FadeIn(yl), run_time=1.2)
    self.play(ShowCreation(curve), FadeIn(form), run_time=0.45 * d)
    self.until(1.0)

    # ---- beat 3: a cloud of noisy poses in the axis-angle ball
    d = self.beat(3)
    plot = VGroup(ax, xl, yl, curve, form)
    self.play(plot.animate.scale(0.42).to_corner(DL, buff=0.35), run_time=1.0)
    fix(plot)
    bc = ORIGIN
    bl = ball().move_to(bc)
    n = 320
    om = rng.normal(size=(n, 3))
    tt = ValueTracker(0)

    def pts(t):
      s_ = float(np.interp(t, np.arange(101), sig))
      return np.array([bc + rot_to_ball(R_gt @ lm.so3_exp(s_ * w)) for w in om])

    cloud = DotCloud(pts(0), radius=0.035).set_color(POSE_C)
    cloud.add_updater(lambda m: m.set_points(pts(tt.get_value())))
    gt_dot = Sphere(radius=0.09).set_color(GT_C).move_to(bc + rot_to_ball(R_gt))
    tdot = Dot(radius=0.06).set_color(WHITE)
    tdot.add_updater(lambda m: m.move_to(ax.i2gp(tt.get_value(), curve)))
    fix(tdot)
    tl = VGroup(T("t=", 34), Integer(0, font_size=34)).arrange(RIGHT, buff=0.1).to_corner(UR, buff=0.6).shift(DOWN * 0.9)
    tl[1].add_updater(lambda m: m.set_value(int(round(tt.get_value()))))
    fix(tl)
    bl_lab = TT(r"rotation = axis $\times$ angle, $|\cdot|\le\pi$", 26, GREY_A).to_corner(DR, buff=0.4)
    fix(bl_lab)
    self.play(FadeIn(bl), FadeIn(gt_dot), FadeIn(cloud), FadeIn(tdot), FadeIn(tl), FadeIn(bl_lab),
              frame.animate.reorient(-20, -12, 0).shift(LEFT * 2.6 + UP * 0.6), eq.animate.scale(0.7).to_corner(UL, buff=0.35), run_time=1.5)
    self.play(tt.animate.set_value(100), run_time=0.62 * d, rate_func=linear)
    self.until(1.0)

    # ---- beat 4: training and the sampler on the group (LiePose's update, as in its code)
    d = self.beat(4)
    self.play(tt.animate.set_value(0), run_time=0.2 * d, rate_func=smooth)
    cloud.clear_updaters()
    tr_eq = T(r"\epsilon_\theta(g_t,\,t,\,I)\ \approx\ \epsilon", 44, t2c={r"\epsilon": NOISE_C, "g_t": POSE_C})
    s1 = T(r"\hat g_0 = g_t\,\mathrm{Exp}(\sigma_t\hat\epsilon)^{-1}", 38, t2c={r"\hat g_0": GT_C, "g_t": POSE_C})
    s2 = T(r"g_{t'} = \hat g_0\,\mathrm{Exp}(c\,\hat\epsilon)\,\mathrm{Exp}(s\,\xi),\ \ \xi\sim\mathcal{N}(0,I_6)", 38,
           t2c={r"\hat g_0": GT_C, "g_{t'}": POSE_C})
    s3 = TT(r"$c, s$ from $\sigma_t,\sigma_{t'}$; 100 steps from a uniform prior", 26, GREY_A)
    col = VGroup(tr_eq, s1, s2, s3).arrange(DOWN, buff=0.32, aligned_edge=LEFT).to_corner(UL, buff=0.45).shift(DOWN * 0.9)
    fix(col)
    self.play(FadeOut(eq), Write(tr_eq), run_time=1.5)
    self.until(0.4)
    self.play(FadeIn(s1, DOWN * 0.2), run_time=1.2)
    self.play(FadeIn(s2, DOWN * 0.2), run_time=1.4)
    self.play(FadeIn(s3), FadeOut(cloud), run_time=1.0)
    self.until(1.0)

    # ---- beat 5: real reverse trajectories (control denoiser, one test crop)
    d = self.beat(5)
    Rt = tr["field_nowm_R"]  # (101 states, 16 samples, 3, 3)
    paths = [np.array([bc + rot_to_ball(Rt[i, j]) for i in range(Rt.shape[0])]) for j in range(Rt.shape[1])]
    sym_dot = Sphere(radius=0.09).set_color(GT_C).move_to(bc + rot_to_ball(R_gt @ S))
    pg = PathGrower(paths, CTRL_C, 2.2, 0.85, head_color=POSE_C)
    cap = caption(r"real run: control denoiser (no world model), 100k$\times$32 checkpoint, one test crop; 16 samples, 100 steps")
    fix(cap)
    self.play(FadeIn(sym_dot), FadeIn(pg.heads), FadeIn(cap), FadeOut(tdot), FadeOut(tl), FadeOut(bl_lab), run_time=1.0)
    self.add(pg.lines)
    self.play(pg.prog.animate.set_value(100), run_time=0.7 * d, rate_func=linear)
    pg.stop()
    self.until(1.0)
    frame.clear_updaters()
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
