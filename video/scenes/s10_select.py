from common import *


class S10(LScene):
  SCENE = "S10"

  def construct(self):
    tr, ry = load("traj.npz"), load("rays.npz")
    uv = ry["field_full_lat_final_uv"]       # (16, n_vertices, 2): mesh vertices at each final LaRC sample, in crop pixels
    Ef = tr["field_full_lat_E_final"]        # energies of the 16 final samples under the field (this crop)

    # ---- beat 1: 16 independent samples
    d = self.beat(1)
    th = 1.25
    emax, emin = Ef.max(), Ef.min()
    cells = Group()
    for i in range(16):
      im = ImageMobject(os.path.join(DATA, "crop.png"), height=th)
      pts = np.array([im.get_corner(UL) + np.array([u / 224 * th, -v / 224 * th, 0]) for u, v in uv[i]])
      dc = DotCloud(pts, radius=0.022).set_color(REN_C).set_opacity(0.55)
      bar = Rectangle(0.16, th * Ef[i] / emax).set_fill(heat(0.25 + 0.75 * (Ef[i] - emin) / (emax - emin)), 1).set_stroke(width=0)
      bar.next_to(im, RIGHT, 0.06).align_to(im, DOWN)
      bar.set_opacity(0)
      cells.add(Group(im, dc, bar))
    cells.arrange_in_grid(2, 8, h_buff=0.22, v_buff=0.35).move_to(UP * 0.7)
    lab = TT(r"$N=16$ samples from independent priors: final poses (CAD vertices projected, display only)", 26, GREY_A)
    lab.next_to(cells, UP, 0.3)
    cap = caption(r"real run: LaRC, 100k$\times$32 checkpoint, one test crop (an example, not a result)")
    self.play(LaggedStart(*[FadeIn(c[:2], scale=0.9) for c in cells], lag_ratio=0.06), FadeIn(lab), FadeIn(cap), run_time=0.6 * d)
    self.until(1.0)

    # ---- beat 2: rank by the energy; consensus as the world-model-free rule
    d = self.beat(2)
    sel = T(r"\hat g=\arg\min_i\ E(g_i;\mathbf z)", 46, REN_C).to_edge(DOWN, buff=1.05)
    self.play(Write(sel), run_time=1.4)
    el = TT(r"bar: $E(g_i;\mathbf z)$", 26, GREY_A).next_to(cells, DOWN, 0.15).align_to(cells, LEFT)
    self.play(*[c[2].animate.set_opacity(1) for c in cells], FadeIn(el), run_time=1.2)
    order = np.argsort(Ef)
    slots = [c.get_center() for c in cells]
    self.play(*[cells[i].animate.move_to(slots[k]) for k, i in enumerate(order)], run_time=1.8)
    best, worst = cells[order[0]][0], cells[order[-1]][0]
    hb = SurroundingRectangle(best, buff=0.04).set_stroke(GT_C, 4)
    hw = SurroundingRectangle(worst, buff=0.04).set_stroke(NOISE_C, 4)
    lb = TT(r"lowest $E$: returned", 24, GT_C).next_to(hb, UP, 0.08)
    lw = TT(r"highest $E$: the sample that went astray", 24, NOISE_C).next_to(hw, DOWN, 0.1).shift(LEFT * 1.6)
    self.play(FadeOut(lab), ShowCreation(hb), FadeIn(lb), run_time=1.0)
    self.until(0.45)
    cons = TT(r"consensus (no world model): the sample with the most others within $10^\circ$", 30, GREY_A).next_to(sel, DOWN, 0.2)
    self.play(FadeIn(cons, UP * 0.2), run_time=1.0)
    self.until(0.75)
    self.play(ShowCreation(hw), FadeIn(lw), run_time=1.0)
    self.until(1.0)
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
