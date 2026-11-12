from common import *

# Every number below is copied from the paper source (see out/REPORT.md for file and line).


def vbar(ax, x, v, color, w=0.5):
  b = Rectangle(w, max(1e-3, ax.c2p(0, v)[1] - ax.c2p(0, 0)[1])).set_fill(color, 0.95).set_stroke(width=0)
  b.move_to(ax.c2p(x, 0), aligned_edge=DOWN)
  return b


class S12(LScene):
  SCENE = "S12"

  def construct(self):
    # ---- beat 1: T-LESS, GT crops, 100k x 32: control vs LaRC (tab:bop rows "Control"/"LaRC" at 100k x 32)
    d = self.beat(1)
    head = TT(r"T-LESS (BOP19, ground-truth crops): recall (\%), 100k steps $\times$ batch 32", 30).to_edge(UP, buff=0.35)
    self.play(FadeIn(head), run_time=1.0)
    ax = small_axes([0, 9, 1], [0, 100, 25], 10.5, 4.6, [25, 50, 75, 100], [], 22).move_to(DOWN * 0.3)
    self.play(ShowCreation(ax), run_time=0.8)
    groups = [("VSD", 21.5, 65.3), ("MSSD", 24.9, 65.5), ("MSPD", 75.2, 86.5), ("AR", 40.5, 72.4)]
    xs = [1.2, 3.2, 5.2, 7.6]
    bars, labs = VGroup(), VGroup()
    for (name, c, l), x in zip(groups, xs):
      bc_, bl_ = vbar(ax, x - 0.3, c, CTRL_C), vbar(ax, x + 0.3, l, REN_C)
      bars.add(bc_, bl_)
      labs.add(T(f"{c:.1f}", 26, CTRL_C).next_to(bc_, UP, 0.08), T(f"{l:.1f}", 26, REN_C).next_to(bl_, UP, 0.08),
               TT(name, 28 if name != "AR" else 32, TXT_C).next_to(ax.c2p(x, 0), DOWN, 0.2))
    leg = VGroup(VGroup(Square(0.25).set_fill(CTRL_C, 1).set_stroke(width=0), TT("control (no world model)", 26)).arrange(RIGHT, buff=0.15),
                 VGroup(Square(0.25).set_fill(REN_C, 1).set_stroke(width=0), TT("LaRC", 26)).arrange(RIGHT, buff=0.15))
    leg.arrange(RIGHT, buff=0.6).next_to(head, DOWN, 0.25)
    self.play(FadeIn(leg), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.12), run_time=0.35 * d)
    self.play(FadeIn(labs), run_time=1.0)
    gain = Arrow(bars[6].get_top() + UP * 0.45, bars[7].get_top() + UP * 0.45, buff=0.05, path_arc=-0.6).set_color(REN_C)
    self.play(ShowCreation(gain), run_time=0.8)
    self.until(1.0)

    # ---- beat 2: rotation / translation accuracies (tab:supp-tless-rt)
    d = self.beat(2)
    self.play(FadeOut(Group(ax, bars, labs, gain, leg)), run_time=0.8)
    rows = [(r"translation error $<2$\,cm", 26.8, 80.3), (r"rotation error $<5^\circ$", 45.7, 71.6)]
    ax2 = small_axes([0, 100, 25], [0, 2, 1], 9.0, 2.6, [], [0, 25, 50, 75, 100], 22).move_to(DOWN * 0.2 + RIGHT * 1.2)
    self.play(ShowCreation(ax2), run_time=0.8)
    for k, (name, a, b) in enumerate(rows):
      y = 1.45 - k * 0.9
      pa, pb = ax2.c2p(a, y), ax2.c2p(b, y)
      da, db = Dot(pa, radius=0.12).set_color(CTRL_C), Dot(pb, radius=0.12).set_color(REN_C)
      ln_ = Line(pa, pb).set_stroke(GREY_B, 5)
      nm = TT(name, 28).next_to(ax2.c2p(0, y), LEFT, 0.3)
      la, lb = T(f"{a:.1f}", 26, CTRL_C).next_to(da, UP, 0.12), T(f"{b:.1f}", 26, REN_C).next_to(db, UP, 0.12)
      self.play(FadeIn(nm), FadeIn(da), FadeIn(la), run_time=0.6)
      self.play(ShowCreation(ln_), TransformFromCopy(da, db), FadeIn(lb), run_time=1.2)
    src = TT(r"share of estimates (\%), control $\to$ LaRC, 100k$\times$32", 24, GREY_A).next_to(ax2, DOWN, 0.5)
    self.play(FadeIn(src), run_time=0.6)
    self.until(1.0)

    # ---- beat 3: LiePose; detector-crop rows kept apart
    d = self.beat(3)
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not head])), run_time=0.8)
    head2 = TT(r"T-LESS average recall (\%)", 32).to_edge(UP, buff=0.35)
    self.play(ReplacementTransform(head, head2), run_time=0.6)
    nl = NumberLine(x_range=[66, 80, 2], width=11, include_numbers=True, decimal_number_config=dict(num_decimal_places=0, font_size=24)).move_to(DOWN * 0.2)
    nl.set_color(GREY_B)
    self.play(ShowCreation(nl), run_time=0.8)
    pts = [(70.1, r"LiePose, published$^\dagger$: 70.1", LIE_C, 1.6), (72.4, r"LaRC (100k$\times$32): 72.4", REN_C, 0.75),
           (73.7, r"LiePose, our training (400k$\times$64): 73.7", LIE_C, 2.3)]
    marks = Group()
    for v, name, c, up in pts:
      dt = Dot(nl.n2p(v), radius=0.12).set_color(c)
      lb = TT(name, 24, c).next_to(dt, UP, up)
      vl = VGroup()
      ln_ = Line(dt.get_center(), lb.get_bottom()).set_stroke(c, 1.5, 0.6)
      marks.add(Group(dt, lb, vl, ln_))
      self.play(FadeIn(marks[-1]), run_time=0.8)
    gt_tag = TT(r"ground-truth crops", 26, GREY_A).next_to(nl, DOWN, 0.7).shift(LEFT * 2.5)
    note = TT(r"$^\dagger$mean of the three published recalls, computed by the authors", 22, GREY_B).next_to(gt_tag, DOWN, 0.1).align_to(gt_tag, LEFT)
    bud = TT(r"LaRC: an eighth of the training images", 26, REN_C).next_to(marks[1][0], DOWN, 0.6)
    self.play(FadeIn(gt_tag), FadeIn(note), FadeIn(bud), run_time=1.0)
    self.until(0.6)
    sep = DashedLine(UP * 2.6 + RIGHT * 4.6, DOWN * 3.4 + RIGHT * 4.6).set_stroke(GREY_B, 2)
    det = Dot(nl.n2p(78.6), radius=0.12).set_color(GREY_B)
    detl = TT(r"GDRNPP 78.6\\detector crops:\\different protocol,\\not comparable", 24, GREY_B).next_to(det, UP, 0.6).shift(RIGHT * 0.2)
    self.play(ShowCreation(sep), FadeIn(det), FadeIn(detl), run_time=1.2)
    self.until(1.0)

    # ---- beat 4: SPEED+ on a log axis (tab:speedplus, 400k x 64 rows and published SPNv3-B)
    d = self.beat(4)
    self.play(FadeOut(Group(*[m for m in self.mobjects])), run_time=0.8)
    head3 = TT(r"SPEED+ (synthetic training only), official score $S$, lower is better", 32).to_edge(UP, buff=0.35)
    self.play(FadeIn(head3), run_time=0.8)
    lo, hi = np.log10(0.03), np.log10(1.2)
    X0, X1 = -5.2, 5.6
    xp = lambda v: X0 + (np.log10(v) - lo) / (hi - lo) * (X1 - X0)
    axl = Line(RIGHT * X0, RIGHT * X1).set_stroke(GREY_B, 2).shift(DOWN * 2.3)
    ticks = VGroup()
    for v in (0.05, 0.1, 0.2, 0.5, 1.0):
      ticks.add(Line(UP * 0.1, DOWN * 0.1).move_to([xp(v), -2.3, 0]).set_stroke(GREY_B, 2),
                T(f"{v:g}", 24, GREY_B).move_to([xp(v), -2.65, 0]))
    logl = TT("log scale", 22, GREY_B).next_to(axl, RIGHT, 0.1).shift(DOWN * 0.35)
    self.play(ShowCreation(axl), FadeIn(ticks), FadeIn(logl), run_time=1.0)
    rows = [("control (no world model)", 0.854, 0.793, CTRL_C), ("LaRC", 0.672, 0.664, REN_C), ("LiePose, our training", 0.234, 0.289, LIE_C),
            ("SPNv3-B (published)", 0.051, 0.079, GREY_A)]
    ys = [1.6, 0.7, -0.2, -1.1]
    for (name, a, b, c), y in zip(rows, ys):
      d1 = Dot([xp(a), y, 0], radius=0.11).set_color(c)
      d2 = Square(0.2).set_fill(c, 1).set_stroke(width=0).move_to([xp(b), y, 0])
      n_ = TT(name, 24, c).next_to(VGroup(d1, d2), UP, 0.12)
      vl = T(f"{a:.3f}\\ /\\ {b:.3f}", 22, c).next_to(VGroup(d1, d2), DOWN, 0.12)
      self.play(FadeIn(d1), FadeIn(d2), FadeIn(n_), FadeIn(vl), run_time=0.9)
    leg = TT(r"$\bullet$ lightbox \quad $\blacksquare$ sunlamp \qquad ours: 400k$\times$64", 24, GREY_A)
    leg.to_edge(DOWN, buff=0.2)
    self.play(FadeIn(leg), run_time=0.8)
    self.until(1.0)

    # ---- beat 5: the claim
    d = self.beat(5)
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)
    c1 = TT(r"large gains over the identical denoiser without the comparison", 38, REN_C)
    c2 = TT(r"not a new state of the art", 38, GREY_A)
    VGroup(c1, c2).arrange(DOWN, buff=0.45)
    self.play(FadeIn(c1, UP * 0.2), run_time=1.0)
    self.play(FadeIn(c2, UP * 0.2), run_time=1.0)
    self.until(1.0)
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
