from common import *


def ln(v):
  v = np.asarray(v, np.float64)
  return (v - v.mean()) / (v.std() + 1e-6)


def strip(v, color, length, height=0.42, vmax=2.5):
  n = len(v)
  g = VGroup(*[Rectangle(length / n, height).set_fill(color, float(np.clip(0.12 + 0.88 * abs(x) / vmax, 0, 1))).set_stroke(width=0)
               for x in v]).arrange(RIGHT, buff=0)
  return g


class S09(L3Scene):
  SCENE = "S09"

  def construct(self):
    tk, tr = load("tokens.npz"), load("traj.npz")
    pca = token_pca()
    frame = self.frame
    R_gt, S = tk["R_gt"], tk["S"]
    z = tk["z"]
    zh0 = tr["field_full_lat_zh0"].astype(np.float64)  # (100 steps, 64, 384): the field's rendering at sample 0's g_t
    R_l, R_c = tr["field_full_lat_R"], tr["field_nowm_R"]

    # ---- beat 1: at step t the frozen field renders at each sample's g_t
    d = self.beat(1)
    zg = TokenGrid([rgb_hex(c) for c in pca(z)], 8, side=0.3, gap=0.03).move_to(LEFT * 4.6 + UP * 1.2)
    zl = T(r"\mathbf z", 34, OBS_C).next_to(zg, UP, 0.15)
    hg = TokenGrid([rgb_hex(c) for c in pca(zh0[0])], 8, side=0.3, gap=0.03).move_to(LEFT * 0.0 + UP * 1.2)
    hl = T(r"\hat{\mathbf z}(g_t)", 34, REN_C).next_to(hg, UP, 0.15)
    gt_ = T(r"g_t", 40, POSE_C).move_to(RIGHT * 4.6 + UP * 1.2)
    arr = Arrow(gt_.get_left(), hg.get_right(), buff=0.3).set_color(REN_C)
    fl = T(r"\mathcal F_\phi\ \text{(frozen)}", 30, REN_C).next_to(arr, UP, 0.1)
    self.play(FadeIn(zg), FadeIn(zl), FadeIn(gt_), run_time=1.0)
    self.play(GrowArrow(arr), FadeIn(fl), run_time=1.0)
    self.play(FadeIn(hg, lag_ratio=0.02), FadeIn(hl), run_time=1.4)
    pe = TT(r"for every sample, at every one of the 100 steps", 28, GREY_A).to_edge(DOWN, buff=0.6)
    self.play(FadeIn(pe), run_time=0.8)
    self.until(1.0)

    # ---- beat 2: one input token per patch (eq. tokens), real values of patch p at the true pose
    d = self.beat(2)
    self.play(FadeOut(Group(arr, fl, gt_, pe)), Group(zg, zl, hg, hl).animate.shift(UP * 0.6), run_time=0.8)
    p = 27
    parts = [(ln(z[p]), OBS_C, 3.3, r"\mathrm{LN}(\mathbf z_p)"), (ln(tk["zh_gt"][p]), REN_C, 3.3, r"\mathrm{LN}(\hat{\mathbf z}_p)"),
             (ln(tk["zh_gt"][p] - z[p]), DIFF_C, 3.3, r"\mathrm{LN}(\hat{\mathbf z}_p-\mathbf z_p)"),
             (np.array([tk["w"][p] / tk["w"].max() * 2.5]), GREY_A, 0.25, r"w_p"), (ln(tk["h_gt"][p]), LAT_C, 1.1, r"\mathrm{LN}(\mathbf h_p)")]
    bar = VGroup()
    labs = VGroup()
    for v, c, Lw, name in parts:
      s_ = strip(v, c, Lw)
      bar.add(s_)
    bar.arrange(RIGHT, buff=0.06).move_to(DOWN * 0.9)
    for s_, (_, c, _, name) in zip(bar, parts):
      labs.add(T(name, 24, c).next_to(s_, DOWN, 0.15))
    dims = T(r"3\cdot384+1+128=1281", 28, GREY_A).next_to(bar, UP, 0.2)
    hlp = VGroup(zg[p].copy().set_stroke(WHITE, 3), hg[p].copy().set_stroke(WHITE, 3))
    self.play(ShowCreation(hlp), run_time=0.6)
    for s_, l_ in zip(bar, labs):
      self.play(FadeIn(s_, lag_ratio=0.002), FadeIn(l_), run_time=0.55)
    self.play(FadeIn(dims), run_time=0.6)
    eq = T(r"x_p=\mathrm{Lin}\big([\,\mathrm{LN}(\mathbf z_p),\ \mathrm{LN}(\hat{\mathbf z}_p),\ \mathrm{LN}(\hat{\mathbf z}_p-\mathbf z_p),\ w_p,\ "
           r"\mathrm{LN}(\mathbf h_p)\,]\big)+e_p", 36).to_edge(DOWN, buff=0.45)
    self.play(Write(eq), run_time=2.0)
    self.until(1.0)

    # ---- beat 3: conditioning token, 4 adaLN blocks, output at c; stop-gradient into the field
    d = self.beat(3)
    self.play(FadeOut(Group(bar, labs, dims, hlp, zg, zl, hg, hl)), eq.animate.scale(0.8).to_edge(UP, buff=0.3), run_time=1.0)
    n = 17  # draw 1 + 16 of the 65 tokens
    xs = np.linspace(-5.6, 5.6, n)
    rows = []
    for li in range(5):
      y = -2.2 + li * 1.15
      row = VGroup(*[Circle(radius=0.13).set_fill(POSE_C if i == 0 else OBS_C, 0.9).set_stroke(WHITE, 1).move_to([x, y, 0])
                     for i, x in enumerate(xs)])
      rows.append(row)
    dots = T(r"\cdots", 30, GREY_B).move_to([xs[-1] + 0.6, -2.2, 0])
    cl = T(r"c=\mathrm{Lin}\big(\mathrm{SiLU}(\gamma_t(t)+\gamma_g(r_t))\big)", 30, POSE_C).next_to(rows[0][0], DOWN, 0.25).shift(RIGHT * 2.2)
    self.play(FadeIn(rows[0], lag_ratio=0.05), FadeIn(dots), FadeIn(cl), run_time=1.4)
    blk = TT(r"4 adaLN transformer blocks, width 256", 26, GREY_A).to_corner(UR, buff=0.4).shift(DOWN * 0.8)
    self.play(FadeIn(blk), run_time=0.6)
    rng = np.random.default_rng(1)
    for li in range(1, 5):
      att = VGroup(*[Line(rows[li - 1][a].get_center(), rows[li][b].get_center()).set_stroke(GREY_B, 0.8, 0.35)
                     for a in range(n) for b in rng.choice(n, 3, replace=False)])
      mod = Arrow(rows[0][0].get_center() + LEFT * 0.4, rows[li][0].get_center() + LEFT * 0.25, buff=0.1, path_arc=0.5,
                  stroke_width=2).set_color(POSE_C)
      self.play(ShowCreation(att, lag_ratio=0.01), FadeIn(rows[li], lag_ratio=0.03), ShowCreation(mod), run_time=0.75)
    out = T(r"\hat\epsilon=\epsilon_\theta\in\mathbb R^6", 34, NOISE_C).next_to(rows[4][0], UP, 0.25)
    self.play(Indicate(rows[4][0], color=NOISE_C, scale_factor=1.6), FadeIn(out), run_time=1.0)
    sg = TT(r"rendering treated as a constant: no gradient reaches $\mathcal F_\phi$", 26, REN_C).to_edge(DOWN, buff=0.25)
    self.play(FadeIn(sg), run_time=0.8)
    self.until(1.0)

    # ---- beat 4: real run: sample 0 of LaRC, the rendering at g_t and the difference shrinking
    d = self.beat(4)
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not frame])), run_time=0.8)
    zg = TokenGrid([rgb_hex(c) for c in pca(z)], 8, side=0.27, gap=0.025).move_to(LEFT * 5.0 + UP * 1.5)
    hg = TokenGrid([rgb_hex(c) for c in pca(zh0[0])], 8, side=0.27, gap=0.025).move_to(LEFT * 2.2 + UP * 1.5)
    dif = lambda i: ((zh0[i] - z) ** 2).mean(-1)
    dmax = max(dif(0).max(), 1e-6)
    dg = TokenGrid([heat(v / dmax) for v in dif(0)], 8, side=0.27, gap=0.025).move_to(LEFT * 3.6 + DOWN * 1.5)
    l1 = T(r"\mathbf z", 30, OBS_C).next_to(zg, UP, 0.12)
    l2 = T(r"\hat{\mathbf z}(g_t)", 30, REN_C).next_to(hg, UP, 0.12)
    l3 = T(r"\|\hat{\mathbf z}_p(g_t)-\mathbf z_p\|^2", 30, DIFF_C).next_to(dg, UP, 0.12)
    fix(zg, hg, dg, l1, l2, l3)
    bc = ORIGIN
    bl = ball(radius=2.2).move_to(bc)
    path = np.array([bc + rot_to_ball(R_l[i, 0]) * 2.2 / BALL_R for i in range(101)])
    gt_d = Group(Sphere(radius=0.08).set_color(GT_C).move_to(bc + rot_to_ball(R_gt) * 2.2 / BALL_R),
                 Sphere(radius=0.08).set_color(GT_C).move_to(bc + rot_to_ball(R_gt @ S) * 2.2 / BALL_R))
    frame.reorient(-20, -10, 0)
    frame.shift(LEFT * 3.2 + DOWN * 0.1)
    st = ValueTracker(0)
    tl = VGroup(T("t=", 30), Integer(100, font_size=30)).arrange(RIGHT, buff=0.08).to_corner(UR, buff=0.5)
    tl[1].add_updater(lambda m: m.set_value(100 - int(round(st.get_value()))))
    fix(tl)
    cap = caption(r"real run: LaRC, 100k$\times$32 checkpoint, one test crop, sample 0; green = ground truth and its symmetric copy (display only)")
    fix(cap)
    pg = PathGrower([path], REN_C, 3.0, 1.0, head_r=0.09, head_color=POSE_C)
    self.play(FadeIn(zg), FadeIn(hg), FadeIn(dg), FadeIn(l1), FadeIn(l2), FadeIn(l3), FadeIn(bl), FadeIn(gt_d), FadeIn(pg.heads),
              FadeIn(tl), FadeIn(cap), run_time=1.2)
    self.add(pg.lines)

    def upd_grids(_):
      i = int(np.clip(round(st.get_value()), 0, 99))
      hg.recolor([rgb_hex(c) for c in pca(zh0[i])])
      dg.recolor([heat(v / dmax) for v in dif(i)])

    upd = Mobject()
    upd.add_updater(upd_grids)
    self.add(upd)
    pg.prog.add_updater(lambda m: m.set_value(st.get_value()))
    self.add(pg.prog)
    self.play(st.animate.set_value(100), run_time=0.75 * d, rate_func=linear)
    upd.clear_updaters()
    pg.prog.clear_updaters()
    pg.stop()
    self.until(1.0)

    # ---- beat 5: LaRC vs control from the same priors and the same noise
    d = self.beat(5)
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not frame])), run_time=0.8)
    frame.reorient(0, 0, 0)
    frame.move_to(ORIGIN)
    r2 = 2.0
    cL, cR = LEFT * 3.4 + DOWN * 0.3, RIGHT * 3.4 + DOWN * 0.3
    bL, bR = ball(radius=r2).move_to(cL), ball(radius=r2).move_to(cR)
    sc = r2 / BALL_R
    paths_l = [np.array([cL + rot_to_ball(R_l[i, j]) * sc for i in range(101)]) for j in range(16)]
    paths_c = [np.array([cR + rot_to_ball(R_c[i, j]) * sc for i in range(101)]) for j in range(16)]
    gts = Group(*[Sphere(radius=0.07).set_color(GT_C).move_to(c + rot_to_ball(R) * sc) for c in (cL, cR) for R in (R_gt, R_gt @ S)])
    tL = TT(r"LaRC (with the world model)", 32, REN_C).next_to(bL, UP, 0.3)
    tR = TT(r"control (without)", 32, CTRL_C).next_to(bR, UP, 0.3)
    same = TT(r"same network, same conditioning, same budget; only the rendered quantities removed", 26, GREY_A).to_edge(UP, buff=0.25)
    cap2 = caption(r"real runs, 100k$\times$32 checkpoints, one test crop; 16 samples from the same priors with the same noise (an example, not a result)")
    fix(cap2)
    gL = PathGrower(paths_l, REN_C, 1.6, 0.75, head_r=0.06, head_color=POSE_C)
    gR = PathGrower(paths_c, CTRL_C, 1.6, 0.75, head_r=0.06, head_color=POSE_C)
    self.play(FadeIn(bL), FadeIn(bR), FadeIn(tL), FadeIn(tR), FadeIn(same), FadeIn(gts), FadeIn(gL.heads), FadeIn(gR.heads),
              FadeIn(cap2), run_time=1.2)
    self.add(gL.lines, gR.lines)
    gR.prog.add_updater(lambda m: m.set_value(gL.prog.get_value()))
    self.add(gR.prog)
    self.play(gL.prog.animate.set_value(100), run_time=0.7 * d, rate_func=linear)
    gR.prog.clear_updaters()
    gL.stop()
    gR.stop()
    self.until(1.0)
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not frame])), run_time=0.8)
    self.finish()
