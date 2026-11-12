from common import *
from PIL import Image


def diverging(v):
  v = float(np.clip(v, -1, 1))
  a, b = np.array([0.31, 0.76, 0.97]), np.array([0.94, 0.38, 0.57])
  return rgb_hex((a if v < 0 else b) * abs(v) + np.array([0.08, 0.09, 0.11]) * (1 - abs(v)))


class S06(L3Scene):
  SCENE = "S06"

  def construct(self):
    tk = load("tokens.npz")
    pca = token_pca()
    img = np.array(Image.open(os.path.join(DATA, "crop.png"))).astype(np.float64) / 255
    frame = self.frame

    # ---- beat 1: the 224 x 224 crop, masked to the visible object
    d = self.beat(1)
    H = 5.2
    cc = LEFT * 3.6 + DOWN * 0.3
    crop = ImageMobject(os.path.join(DATA, "crop.png"), height=H).move_to(cc)
    box = SurroundingRectangle(crop, buff=0).set_stroke(GREY_B, 1.5)
    size = TT(r"$224\times224$ crop $I$", 28, OBS_C).next_to(box, UP, 0.15)
    self.play(FadeIn(crop), ShowCreation(box), FadeIn(size), run_time=1.2)
    m = np.array(Image.open(os.path.join(DATA, "mask.png"))) > 127
    ov = np.zeros((224, 224, 4), np.uint8)
    ov[m] = (79, 195, 247, 110)
    Image.fromarray(ov).save("/tmp/claude_mask_ov.png") if False else None
    ovp = os.path.join(DATA, "mask_ov.png")
    Image.fromarray(ov).save(ovp)
    mo = ImageMobject(ovp, height=H).move_to(cc)
    ml = TT(r"visible mask: pixels outside set to zero", 26, OBS_C).next_to(box, DOWN, 0.15)
    self.play(FadeIn(mo), FadeIn(ml), run_time=1.2)
    self.play(FadeOut(mo), run_time=1.0)
    self.until(1.0)

    # ---- beat 2: 14-px patches -> 16 x 16 -> DINOv2 tokens -> 2 x 2 pooled 8 x 8 (real tokens, PCA colours)
    d = self.beat(2)
    step = H / 16
    lines = VGroup(*[Line(box.get_corner(UL) + RIGHT * i * step, box.get_corner(DL) + RIGHT * i * step) for i in range(17)],
                   *[Line(box.get_corner(UL) + DOWN * i * step, box.get_corner(UR) + DOWN * i * step) for i in range(17)])
    lines.set_stroke(WHITE, 0.8, 0.5)
    pl = TT(r"$16\times16$ patches of $14\times14$ px", 26, GREY_A).next_to(box, DOWN, 0.15)
    self.play(ShowCreation(lines, lag_ratio=0.02), FadeOut(ml), FadeIn(pl), run_time=1.6)
    patches = VGroup()
    for i in range(16):
      for j in range(16):
        c = img[i * 14:(i + 1) * 14, j * 14:(j + 1) * 14].mean((0, 1))
        sq = Square(step).set_fill(rgb_hex(c), 1).set_stroke(BG, 0.5)
        sq.move_to(box.get_corner(UL) + RIGHT * (j + 0.5) * step + DOWN * (i + 0.5) * step)
        patches.add(sq)
    self.add(patches)
    dino = TT(r"frozen DINOv2 ViT-S/14", 30, OBS_C).move_to(UP * 3.3 + RIGHT * 0.6)
    tgt = RIGHT * 2.6 + DOWN * 0.3
    small = 0.24
    self.play(FadeIn(dino), patches.animate.arrange_in_grid(16, 16, buff=0.02).set_width(16 * small).move_to(tgt), run_time=2.0)
    p16 = TT(r"$16\times16$ tokens, 384-d each", 26, GREY_A).next_to(patches, DOWN, 0.15)
    self.play(FadeIn(p16), run_time=0.6)
    cols = [rgb_hex(c) for c in pca(tk["z"])]
    grid = TokenGrid(cols, 8, side=2 * small * 0.98, gap=0.04).move_to(tgt)
    groups = [VGroup(*[patches[(2 * i + a) * 16 + 2 * j + b] for a in (0, 1) for b in (0, 1)]) for i in range(8) for j in range(8)]
    self.play(*[ReplacementTransform(g, sq) for g, sq in zip(groups, grid)], FadeOut(p16), run_time=1.6)
    zl = T(r"\mathbf{z}\in\mathbb{R}^{64\times384}", 40, OBS_C).next_to(grid, DOWN, 0.25)
    pool = TT(r"average-pooled $2\times2$ $\to$ $8\times8$; colours: first 3 principal components", 22, GREY_B).next_to(zl, DOWN, 0.12)
    self.play(Write(zl), FadeIn(pool), run_time=1.2)
    # one token as its 384 real values
    p = 27
    zp = tk["z"][p]
    zs = zp / (np.abs(zp).max() + 1e-6)
    strip = VGroup(*[Rectangle(0.42, 4.6 / 384).set_fill(diverging(v), 1).set_stroke(width=0) for v in zs]).arrange(DOWN, buff=0)
    strip.move_to(RIGHT * 6.3 + DOWN * 0.3)
    sl = T(r"\mathbf{z}_p\in\mathbb{R}^{384}", 30, OBS_C).next_to(strip, UP, 0.15)
    hl = grid[p].copy().set_stroke(WHITE, 3)
    self.play(ShowCreation(hl), run_time=0.6)
    self.play(TransformFromCopy(grid[p], strip), FadeIn(sl), run_time=1.4)
    self.until(1.0)

    # ---- beat 3: per-token weights (real, from the token cache)
    d = self.beat(3)
    self.play(FadeOut(Group(strip, sl, hl, lines, pool, dino)), run_time=0.8)
    w = tk["w"]
    wg = TokenGrid([heat(v / w.max()) for v in w], 8, side=2 * small * 0.98, gap=0.04).move_to(RIGHT * 2.6 + DOWN * 0.3)
    self.play(grid.animate.shift(LEFT * 0.01), zl.animate.shift(LEFT * 0.01), run_time=0.2)
    self.play(FadeOut(zl), Transform(grid.copy(), wg), run_time=1.4)
    weq = T(r"w_p = w_0 + m_p,\qquad w_0 = 0.05", 40).next_to(wg, UP, 0.3)
    wl = TT(r"$m_p$: visible mask, max-pooled to the token grid, dilated by one token", 24, GREY_A).next_to(wg, DOWN, 0.25)
    self.play(Write(weq), run_time=1.4)
    self.play(FadeIn(wl), run_time=0.8)
    self.until(1.0)

    # ---- beat 4: the crop camera k (real values of this crop) and rays through the crop
    d = self.beat(4)
    s, x0, y0, bw, bh, fr, fx, fy, cx, cy = [float(v) for v in tk["k"]]
    self.play(FadeOut(Group(*[m_ for m_ in self.mobjects if m_ not in (crop, box)])), run_time=0.8)
    kk = T(r"k = \big(\text{zoom},\ \text{crop origin},\ \text{box size},\ \text{intrinsics } K\big)", 34).to_edge(UP, buff=0.35)
    fix(kk)
    self.play(Write(kk), run_time=1.4)
    # 3-D: camera centre C, image plane at unit depth, the crop on it, and rays K^-1 [u, v, 1] through the crop's corners.
    # camera axes -> world: x (image right) -> OUT, y (image down) -> DOWN, z (forward) -> RIGHT
    side = 224 / s
    C = LEFT * 4.6 + DOWN * 0.4
    L = 5.0
    to_w = lambda X, Y, Z: C + L * (X * OUT + Y * DOWN + Z * RIGHT)
    pix = lambda u, v, Z=1.0: to_w((u - cx) / fx * Z, (v - cy) / fy * Z, Z)
    plane = Polygon(pix(0, 0), pix(720, 0), pix(720, 540), pix(0, 540)).set_stroke(GREY_B, 2).set_fill(GREY_E, 0.25)
    cr = [pix(x0, y0), pix(x0 + side, y0), pix(x0 + side, y0 + side), pix(x0, y0 + side)]
    crop_q = Polygon(*cr).set_stroke(OBS_C, 3).set_fill(OBS_C, 0.25)
    rays = VGroup(*[Line(C, C + 2.4 * (q - C)).set_stroke(OBS_C, 2, 0.85) for q in cr])
    cam_dot = Sphere(radius=0.09).set_color(WHITE).move_to(C)
    cam_l = TT("camera centre", 24, GREY_A)
    pl_l = TT(r"image plane ($K$)", 24, GREY_B)
    cr_l = TT(r"crop (origin, size, zoom)", 24, OBS_C)
    fix(cam_l, pl_l, cr_l)
    self.play(crop.animate.scale(0.5).to_corner(DR, buff=0.4).shift(UP * 0.5), box.animate.scale(0.5).to_corner(DR, buff=0.4).shift(UP * 0.5), run_time=1.0)
    fix(crop, box)
    self.play(frame.animate.reorient(-38, 62, 0).move_to(C + RIGHT * 4.5), run_time=1.6)
    self.play(FadeIn(cam_dot), ShowCreation(plane), run_time=1.0)
    self.play(ShowCreation(crop_q), *[ShowCreation(r) for r in rays], run_time=1.6)
    cam_l.move_to(DOWN * 2.1 + LEFT * 4.4)
    pl_l.move_to(UP * 2.4 + LEFT * 0.6)
    cr_l.move_to(UP * 2.4 + RIGHT * 3.6)
    self.play(FadeIn(cam_l), FadeIn(pl_l), FadeIn(cr_l), run_time=0.8)
    nopose = TT(r"$k$ holds no pose information", 28, GREY_A).to_corner(DL, buff=0.5)
    fix(nopose)
    self.play(FadeIn(nopose), run_time=0.8)
    self.until(1.0)
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
