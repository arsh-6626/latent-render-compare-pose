from common import *
from PIL import Image

# Numbers: tab:wm-disc and the analysis paragraph of paper/sec/5_experiments.tex (see out/REPORT.md).


class S13(LScene):
  SCENE = "S13"

  def construct(self):
    L = load("landscape.npz")
    E = L["E"]
    En = (E - E.min()) / (E.max() - E.min())
    img = np.array([[np.array([int(heat(v)[i:i + 2], 16) for i in (1, 3, 5)]) for v in row] for row in En[::-1]], np.uint8)
    p = os.path.join(DATA, "landscape_top.png")
    Image.fromarray(img).resize((73 * 8, 37 * 8), Image.NEAREST).save(p)

    # ---- beat 1: P(E(perturbed) > E(true)) by perturbation size, SPEED+
    d = self.beat(1)
    hm = ImageMobject(p, height=2.6).to_corner(UL, buff=0.5).shift(DOWN * 0.5)
    W, H = hm.get_width(), hm.get_height()
    deg = W / 360
    gt = hm.get_center()
    rings = VGroup(*[Circle(radius=r * deg).move_to(gt).set_stroke(c, 2.5) for r, c in ((3, WHITE), (10, YELLOW), (30, NOISE_C))])
    hl = TT(r"perturb the true pose by $1$--$3^\circ$, $3$--$10^\circ$, $10$--$30^\circ$ \\ (rings: sizes on the landscape slice)", 24, GREY_A)
    hl.next_to(hm, DOWN, 0.15).align_to(hm, LEFT)
    q = T(r"P\big(E(g)>E(g^\star)\big)", 40).to_corner(UR, buff=0.5).shift(LEFT * 0.8)
    self.play(FadeIn(hm), FadeIn(hl), run_time=1.0)
    self.play(ShowCreation(rings, lag_ratio=0.4), Write(q), run_time=1.6)
    ax0 = small_axes([0, 3, 1], [0, 0.6, 0.1], 5.8, 3.4, [], [], 22).to_corner(DR, buff=0.7).shift(UP * 0.5 + RIGHT * 0.2)
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
      ax0.add(T(f"{v:.1f}", 22, GREY_B).next_to(ax0.c2p(0, v - 0.4), LEFT, 0.12))
    class _A:
      def c2p(self_, x, y):
        return ax0.c2p(x, y - 0.4)
    ax = ax0
    axp = _A()
    bins = [r"$1$--$3^\circ$", r"$3$--$10^\circ$", r"$10$--$30^\circ$"]
    bl = VGroup(*[TT(b, 24).next_to(axp.c2p(i + 0.5, 0.4), DOWN, 0.15) for i, b in enumerate(bins)])
    chance = DashedLine(axp.c2p(0, 0.5), axp.c2p(3, 0.5)).set_stroke(GREY_B, 2)
    cl = TT("chance", 22, GREY_B).next_to(chance.get_start(), UP, 0.06).shift(RIGHT * 0.45)
    self.play(ShowCreation(ax), FadeIn(bl), ShowCreation(chance), FadeIn(cl), run_time=1.2)
    sp = [0.65, 0.84, 0.98]
    spl = VMobject().set_points_as_corners([axp.c2p(i + 0.5, v) for i, v in enumerate(sp)]).set_stroke(TEAL_B, 4)
    spd = VGroup(*[Dot(axp.c2p(i + 0.5, v), radius=0.09).set_color(TEAL_B) for i, v in enumerate(sp)])
    spv = VGroup(*[T(f"{v:.2f}", 24, TEAL_B).next_to(dd, UP, 0.1) for v, dd in zip(sp, spd)])
    spn = TT(r"SPEED+, field", 26, TEAL_B).next_to(q, DOWN, 0.2).align_to(q, LEFT)
    self.play(ShowCreation(spl), FadeIn(spd), FadeIn(spv), FadeIn(spn), run_time=1.6)
    lt1 = TT(r"below $1^\circ$: 0.52 (chance)", 24, TEAL_B).next_to(chance.get_end(), DOWN, 0.12).shift(LEFT * 1.0)
    self.until(0.8)
    self.play(FadeIn(lt1), run_time=0.8)
    self.until(1.0)

    # ---- beat 2: T-LESS: weaker; gross errors
    d = self.beat(2)
    tl = [0.53, 0.66, 0.86]
    tll = VMobject().set_points_as_corners([axp.c2p(i + 0.5, v) for i, v in enumerate(tl)]).set_stroke(REN_C, 4)
    tld = VGroup(*[Dot(axp.c2p(i + 0.5, v), radius=0.09).set_color(REN_C) for i, v in enumerate(tl)])
    tlv = VGroup(*[T(f"{v:.2f}", 24, REN_C).next_to(dd, DOWN, 0.12) for v, dd in zip(tl, tld)])
    tln = TT(r"T-LESS, field", 26, REN_C).next_to(spn, RIGHT, 0.5)
    self.play(ShowCreation(tll), FadeIn(tld), FadeIn(tlv), FadeIn(tln), run_time=1.6)
    pick = TT(r"pose picked by the energy among candidates (T-LESS):\\ median error $3.4^\circ$, mean $14.2^\circ$: gross errors", 28)
    pick.next_to(hl, DOWN, 0.5).align_to(hl, LEFT)
    self.until(0.45)
    self.play(FadeIn(pick, UP * 0.2), run_time=1.2)
    src = caption(r"held-out observations: T-LESS 500 held-out training crops; SPEED+ 300 crops of the synthetic validation split")
    self.play(FadeIn(src), run_time=0.8)
    self.until(1.0)

    # ---- beat 3: the symmetric tie
    d = self.beat(3)
    self.play(FadeOut(rings), FadeOut(lt1), run_time=0.6)
    sym = VGroup(*[Circle(radius=0.12).move_to(gt + RIGHT * a * deg).set_stroke(GT_C, 3) for a in (0, -180, 180)])
    tie = TT(r"true pose and its $180^\circ$ copy: the same image, a tie that BOP's symmetric metrics count as correct", 24, GT_C)
    tie.next_to(pick, DOWN, 0.3).align_to(pick, LEFT)
    self.play(ShowCreation(sym), FadeIn(tie), run_time=1.4)
    self.until(1.0)
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
