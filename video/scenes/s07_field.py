from common import *

BOX = 2.0  # display half-size of the object box (r_o)


def plane_image(name, side=2 * BOX):
  return ImageMobject(os.path.join(DATA, f"plane_{name}_disp.png"), height=side)


class S07(L3Scene):
  SCENE = "S07"

  def construct(self):
    tk, ry, L = load("tokens.npz"), load("rays.npz"), load("landscape.npz")
    pca = token_pca()
    frame = self.frame
    r_o = float(ry["r_o"])
    X = ry["gt_X"]          # (256 rays, 32 samples, 3) sample points in the object frame / r_o, at the GT pose
    sig, al, Tr, w = ry["gt_sigma"], ry["gt_alpha"], ry["gt_T"], ry["gt_w"]
    RAY = 136               # the ray through sub-patch (8, 8), centre of the crop

    # ---- beat 1: the question the world model answers
    d = self.beat(1)
    q = T(r"\big(\hat{\mathbf z}(g),\ o(g),\ \mathbf h(g)\big)=\mathcal{F}_\phi(g,\ k,\ a)", 46,
          t2c={r"\hat{\mathbf z}(g)": REN_C, r"\mathbf h(g)": LAT_C, r"\mathcal{F}_\phi": REN_C}).to_edge(UP, buff=0.35)
    fix(q)
    zg = TokenGrid([rgb_hex(c) for c in pca(tk["z"])], 8, side=0.3, gap=0.03).move_to(LEFT * 4.5 + DOWN * 0.4)
    zgl = T(r"\mathbf z\ \text{(observed)}", 30, OBS_C).next_to(zg, DOWN, 0.2)
    qg = TokenGrid(["#22262e"] * 64, 8, side=0.3, gap=0.03).move_to(RIGHT * 4.5 + DOWN * 0.4)
    qm = T("?", 90, REN_C).move_to(qg)
    qgl = T(r"\hat{\mathbf z}(g)\ \text{at any pose } g", 30, REN_C).next_to(qg, DOWN, 0.2)
    fix(zg, zgl, qg, qm, qgl)
    self.play(Write(q), FadeIn(zg), FadeIn(zgl), run_time=1.6)
    self.play(FadeIn(qg), FadeIn(qm), FadeIn(qgl), run_time=1.0)
    self.until(1.0)

    # ---- beat 2: the box and the tri-planes (real learned features, PCA colours)
    d = self.beat(2)
    self.play(FadeOut(Group(zg, zgl, qg, qm, qgl)), run_time=0.6)
    cube = VGroup(*[Line(BOX * np.array(a), BOX * np.array(b)) for a, b in
                    [((-1, -1, -1), (1, -1, -1)), ((-1, 1, -1), (1, 1, -1)), ((-1, -1, 1), (1, -1, 1)), ((-1, 1, 1), (1, 1, 1)),
                     ((-1, -1, -1), (-1, 1, -1)), ((1, -1, -1), (1, 1, -1)), ((-1, -1, 1), (-1, 1, 1)), ((1, -1, 1), (1, 1, 1)),
                     ((-1, -1, -1), (-1, -1, 1)), ((1, -1, -1), (1, -1, 1)), ((-1, 1, -1), (-1, 1, 1)), ((1, 1, -1), (1, 1, 1))]])
    cube.set_stroke(GREY_A, 1.5, 0.8)
    mesh = mesh5(scale=BOX / r_o)  # the CAD model in the object frame, same scale as the box
    frame.reorient(30, 65, 0)
    frame.set_height(9.5)
    self.play(ShowCreation(cube), FadeIn(mesh), run_time=1.6)
    rl = TT(r"box half-size $r_o = 1.2\times$ object radius: the only number taken from the CAD model", 26, GREY_A).to_edge(DOWN, buff=0.3)
    fix(rl)
    self.play(FadeIn(rl), run_time=0.8)
    self.wait(1.0)
    self.play(FadeOut(mesh), run_time=1.0)
    pxy = plane_image("xy").move_to(IN * BOX)
    pxz = plane_image("xz").rotate(PI / 2, RIGHT).move_to(DOWN * BOX)
    pyz = plane_image("yz").rotate(PI / 2, RIGHT).rotate(PI / 2, OUT).move_to(LEFT * BOX)
    planes = Group(pxy, pxz, pyz)
    for pmob in planes:
      pmob.set_opacity(0.85)
    self.play(LaggedStart(FadeIn(pxy), FadeIn(pxz), FadeIn(pyz), lag_ratio=0.4), run_time=2.4)
    pl = TT(r"tri-planes $xy,\ xz,\ yz$: $128\times128$, 32 channels (learned; first 3 principal components)", 26, LAT_C)
    pl.to_edge(DOWN, buff=0.3)
    fix(pl)
    self.play(FadeOut(rl), FadeIn(pl), run_time=0.8)
    frame.add_updater(lambda m, dt: m.increment_theta(0.08 * dt))
    self.until(1.0)
    frame.clear_updaters()

    # ---- beat 3: a point x -> three bilinear samples, summed -> MLP -> sigma(x), h(x)
    d = self.beat(3)
    frame.clear_updaters()
    scr_r = frame.get_orientation().apply(RIGHT)
    self.play(frame.animate.set_height(11).shift(-2.6 * scr_r), run_time=1.2)
    j_hi = int(np.argmax(w[RAY]))
    x = BOX * X[RAY, j_hi]
    xd = Sphere(radius=0.08).set_color(POSE_C).move_to(x)
    xl = T("x", 34, POSE_C)
    xl.add_updater(lambda m: m.move_to(frame.to_fixed_frame_point(x) + UR * 0.25) if False else None)
    proj = [np.array([x[0], x[1], -BOX]), np.array([x[0], -BOX, x[2]]), np.array([-BOX, x[1], x[2]])]
    dl = VGroup(*[DashedLine(x, p_, dash_length=0.08).set_stroke(POSE_C, 2) for p_ in proj])
    pd = Group(*[Sphere(radius=0.06).set_color(WHITE).move_to(p_) for p_ in proj])
    self.play(FadeIn(xd, scale=0.3), run_time=0.8)
    self.play(ShowCreation(dl), FadeIn(pd), run_time=1.4)
    f1 = T(r"f(x)=\sum_{P\in\{xy,xz,yz\}}\mathrm{bilinear}\big(P,\ \pi_P(x)\big)", 34, LAT_C)
    f2 = T(r"\big(\sigma(x),\ \mathbf h(x)\big)=\mathrm{MLP}\big([\,f(x),\ \gamma(x)\,]\big),\quad \mathbf h(x)\in\mathbb R^{128}", 34)
    f3 = TT(r"two layers, width 128; $\sigma(x)=0$ outside the box", 26, GREY_A)
    col = VGroup(f1, f2, f3).arrange(DOWN, buff=0.25, aligned_edge=LEFT).to_corner(UL, buff=0.35).shift(DOWN * 0.9)
    fix(col)
    self.play(FadeOut(q), Write(f1), run_time=1.6)
    self.play(Write(f2), run_time=1.8)
    self.play(FadeIn(f3), run_time=0.8)
    self.until(1.0)

    # ---- beat 4: rays through 16 x 16 sub-patch centres, 32 samples each (real geometry at the GT pose, object frame)
    d = self.beat(4)
    self.play(FadeOut(Group(dl, pd, col)), run_time=0.6)
    seg = [(BOX * (X[i, 0] + 1.6 * (X[i, 0] - X[i, -1])), BOX * (X[i, -1] - 0.6 * (X[i, 0] - X[i, -1]))) for i in range(256)]
    rays = VGroup(*[Line(a, b) for a, b in seg]).set_stroke(OBS_C, 0.8, 0.45)
    pts = BOX * X.reshape(-1, 3)
    samp = DotCloud(pts, radius=0.018).set_color(WHITE).set_opacity(0.55)
    self.play(ShowCreation(rays, lag_ratio=0.002), run_time=2.2)
    self.play(FadeIn(samp), FadeOut(xd), run_time=1.4)
    rl2 = TT(r"$16\times16$ rays through the crop, 32 samples within $\pm r_o$ of the object's depth", 26, OBS_C).to_edge(DOWN, buff=0.3)
    fix(rl2)
    self.play(FadeOut(pl), FadeIn(rl2), run_time=0.8)
    self.until(1.0)

    # ---- beat 5: one real ray: density -> alpha -> transmittance -> weights
    d = self.beat(5)
    scr_r = frame.get_orientation().apply(RIGHT)
    self.play(frame.animate.shift(5.2 * scr_r), run_time=1.0)
    hi = Line(*seg[RAY]).set_stroke(POSE_C, 4)
    self.play(ShowCreation(hi), rays.animate.set_stroke(opacity=0.12), samp.animate.set_opacity(0.15), run_time=1.0)
    # colour the samples of this ray by their weight
    wr = w[RAY]
    rs = DotCloud(BOX * X[RAY], radius=0.05).set_color(POSE_C)
    self.play(FadeIn(rs), run_time=0.6)
    ax = small_axes([0, 32, 8], [0, 1, 0.5], 5.6, 2.2, [0.5, 1], [0, 8, 16, 24, 32], 20).to_corner(UR, buff=0.4).shift(DOWN * 0.3)
    jl = TT(r"sample $j$ along the ray", 20, GREY_B).next_to(ax, DOWN, 0.05)
    fix(ax, jl)
    smax = sig[RAY].max()
    bars_s = VGroup(*[Rectangle(5.6 / 32 * 0.8, max(1e-3, 2.2 * v / smax)).set_fill(GREY_B, 0.9).set_stroke(width=0)
                      .move_to(ax.c2p(j + 0.5, 0), aligned_edge=DOWN) for j, v in enumerate(sig[RAY])])
    lab_s = T(r"\sigma_j\ (\text{scaled})", 28, GREY_B).next_to(ax, UP, 0.1).align_to(ax, LEFT)
    fix(bars_s, lab_s)
    self.play(FadeIn(ax), FadeIn(jl), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars_s], lag_ratio=0.03), FadeIn(lab_s), run_time=2.0)
    e1 = T(r"\alpha_j = 1-e^{-\sigma_j\delta}", 34, TXT_C)
    e2 = T(r"T_j=\prod_{i<j}(1-\alpha_i)", 34, TEAL_B)
    e3 = T(r"w_j=T_j\,\alpha_j", 34, POSE_C)
    ecol = VGroup(e1, e2, e3).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(ax, DOWN, 0.55).align_to(ax, LEFT)
    fix(ecol)
    self.play(Write(e1), run_time=1.2)
    Tl = VMobject().set_points_as_corners([ax.c2p(j + 0.5, v) for j, v in enumerate(Tr[RAY])]).set_stroke(TEAL_B, 3)
    fix(Tl)
    self.play(Write(e2), ShowCreation(Tl), run_time=1.8)
    bars_w = VGroup(*[Rectangle(5.6 / 32 * 0.8, max(1e-3, 2.2 * v)).set_fill(POSE_C, 0.95).set_stroke(width=0)
                      .move_to(ax.c2p(j + 0.5, 0), aligned_edge=DOWN) for j, v in enumerate(wr)])
    fix(bars_w)
    self.play(Write(e3), ReplacementTransform(bars_s, bars_w), FadeOut(lab_s), run_time=1.8)
    rs.set_radii([0.03 + 0.12 * v / wr.max() for v in wr])
    self.play(Indicate(rs, scale_factor=1.0, color=WHITE), run_time=1.0)
    self.until(1.0)

    # ---- beat 6: opacity and the latent of the surface; head -> token feature
    d = self.beat(6)
    o_eq = T(r"o=\sum_j w_j,\qquad \bar{\mathbf h}=\frac{1}{o}\sum_j w_j\,\mathbf h(x_j)", 36, t2c={r"\bar{\mathbf h}": LAT_C})
    o_eq.next_to(ecol, DOWN, 0.4).align_to(ecol, LEFT)
    fix(o_eq)
    self.play(Write(o_eq), run_time=1.6)
    og = TokenGrid([rgb_hex([v, v, v]) for v in ry["gt_opa16"]], 16, side=0.17, gap=0.012).to_corner(DL, buff=0.5).shift(UP * 0.3)
    ogl = TT(r"opacity $o$ of all $16\times16$ rays", 24, GREY_A).next_to(og, UP, 0.12)
    fix(og, ogl)
    self.play(FadeOut(rl2), FadeIn(ogl), LaggedStart(*[FadeIn(sq) for sq in og], lag_ratio=0.004), run_time=2.0)
    hd = TT(r"head$(\bar{\mathbf h},\ \text{view direction})\to$ token feature; rays that miss: background feature", 26, REN_C)
    hd.to_edge(DOWN, buff=0.3)
    fix(hd)
    self.play(FadeIn(hd), run_time=1.0)
    self.until(1.0)

    # ---- beat 7: the appearance code of the observation
    d = self.beat(7)
    self.play(FadeOut(Group(ax, jl, bars_w, Tl, ecol, hi, rs)), run_time=0.8)
    zg2 = TokenGrid([rgb_hex(c) for c in pca(tk["z"])], 8, side=0.26, gap=0.025).to_corner(UR, buff=0.5).shift(DOWN * 0.4)
    fix(zg2)
    zm, zs = tk["z"].mean(0), tk["z"].std(0)
    def strip(v, c):
      v = (v - v.min()) / (np.ptp(v) + 1e-9)
      return VGroup(*[Rectangle(2.2 / 96, 0.32).set_fill(c, float(0.15 + 0.85 * x)).set_stroke(width=0) for x in v[:96]]).arrange(RIGHT, buff=0)
    sm, ss = strip(zm, OBS_C), strip(zs, OBS_C)
    al_ = VGroup(VGroup(T(r"\text{mean}", 24), sm).arrange(RIGHT, buff=0.15), VGroup(T(r"\text{std}", 24), ss).arrange(RIGHT, buff=0.15))
    al_.arrange(DOWN, buff=0.12, aligned_edge=RIGHT).next_to(zg2, DOWN, 0.35).to_edge(RIGHT, buff=0.5)
    a_l = T(r"a=\mathrm{app}\big(\mathrm{mean}_p\mathbf z_p,\ \mathrm{std}_p\mathbf z_p\big)", 32, OBS_C).next_to(al_, DOWN, 0.25).to_edge(RIGHT, buff=0.5)
    fix(al_, a_l)
    self.play(FadeIn(zg2), run_time=0.8)
    self.play(TransformFromCopy(zg2, al_), run_time=1.4)
    self.play(Write(a_l), run_time=1.2)
    film = T(r"\hat f = o\cdot\mathrm{head}\big((1+s(a))\odot\bar{\mathbf h}+b(a),\ \dots\big)+(1-o)\cdot\mathrm{bg}(a)", 32).to_edge(DOWN, buff=0.3)
    fix(film)
    self.play(FadeOut(hd), Write(film), run_time=1.8)
    same = TT(r"same $a$ for every hypothesis $g$:\\ it cannot rank poses", 26, GREY_A).next_to(a_l, DOWN, 0.25).to_edge(RIGHT, buff=0.5)
    fix(same)
    self.play(FadeIn(same), run_time=0.8)
    self.until(1.0)

    # ---- beat 8: pooled outputs: observed vs rendered at the true pose vs 40 degrees off
    d = self.beat(8)
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not frame])), run_time=0.8)
    frame.reorient(0, 0, 0)
    frame.set_height(8).move_to(ORIGIN)
    side, gap = 0.36, 0.035
    G = [TokenGrid([rgb_hex(c) for c in pca(tk[k_])], 8, side=side, gap=gap) for k_ in ("z", "zh_gt", "zh_wr")]
    xs = [-4.4, 0, 4.4]
    labs = [T(r"\mathbf z\ \text{observed}", 32, OBS_C), T(r"\hat{\mathbf z}(g^\star)\ \text{true pose}", 32, REN_C),
            T(r"\hat{\mathbf z}(g)\ \text{40}^\circ\text{ off}", 32, REN_C)]
    for g_, x_, l_ in zip(G, xs, labs):
      g_.move_to(RIGHT * x_ + UP * 1.3)
      l_.next_to(g_, UP, 0.2)
    wmeq = T(r"\big(\hat{\mathbf z}(g),\ o(g),\ \mathbf h(g)\big)\in\mathbb R^{64\times384}\times\mathbb R^{64}\times\mathbb R^{64\times128}", 34)
    wmeq.to_edge(DOWN, buff=0.5)
    pool = TT(r"pooled $2\times2$ to the $8\times8$ grid of the observation", 26, GREY_A).next_to(wmeq, UP, 0.2)
    self.play(FadeIn(G[0]), FadeIn(labs[0]), run_time=1.0)
    self.play(FadeIn(G[1], lag_ratio=0.02), FadeIn(labs[1]), run_time=1.6)
    self.play(FadeIn(G[2], lag_ratio=0.02), FadeIn(labs[2]), run_time=1.6)
    self.play(FadeIn(pool), Write(wmeq), run_time=1.8)
    self.until(1.0)

    # ---- beat 9: the energy
    d = self.beat(9)
    self.play(FadeOut(pool), FadeOut(wmeq), run_time=0.5)
    en = T(r"E(g;\mathbf z)=\frac{\sum_p w_p\,\frac{1}{D}\,\|\hat{\mathbf z}_p(g)-\mathbf z_p\|^2}{\sum_p w_p},\qquad D=384", 40,
           t2c={r"\hat{\mathbf z}_p(g)": REN_C, r"\mathbf z_p": OBS_C}).scale(0.85).to_edge(DOWN, buff=0.25)
    self.play(Write(en), run_time=1.8)
    dmax = max(tk["d_gt"].max(), tk["d_wr"].max())
    D1 = TokenGrid([heat(v / dmax) for v in tk["d_gt"] * tk["w"] / tk["w"].max()], 8, side=0.19, gap=0.02).next_to(G[1], DOWN, 0.25)
    D2 = TokenGrid([heat(v / dmax) for v in tk["d_wr"] * tk["w"] / tk["w"].max()], 8, side=0.19, gap=0.02).next_to(G[2], DOWN, 0.25)
    dl_ = TT(r"$w_p\,\|\hat{\mathbf z}_p-\mathbf z_p\|^2$", 24, DIFF_C).next_to(D1, LEFT, 0.4)
    self.play(FadeIn(D1, lag_ratio=0.02), FadeIn(D2, lag_ratio=0.02), FadeIn(dl_), run_time=1.6)
    Eg, Ew = float(tk["E_gt"]), float(tk["E_wr"])
    b1 = Rectangle(0.3, 1.5 * Eg / Ew).set_fill(GT_C, 0.9).set_stroke(width=0).next_to(D1, RIGHT, 0.3).align_to(D1, DOWN)
    b2 = Rectangle(0.3, 1.5).set_fill(NOISE_C, 0.9).set_stroke(width=0).next_to(D2, RIGHT, 0.3).align_to(D2, DOWN)
    bl_ = VGroup(T("E", 26, GT_C).next_to(b1, UP, 0.08), T("E", 26, NOISE_C).next_to(b2, UP, 0.08))
    self.play(GrowFromEdge(b1, DOWN), GrowFromEdge(b2, DOWN), FadeIn(bl_), run_time=1.2)
    self.until(1.0)

    # ---- beat 10: the energy landscape on a slice of SO(3) (real field, this crop)
    d = self.beat(10)
    eL = T(r"E(g;\mathbf z)", 44).to_corner(UL, buff=0.4)
    fix(eL)
    self.play(FadeOut(Group(*G, *labs, D1, D2, dl_, b1, b2, bl_, en)), FadeIn(eL), run_time=1.0)
    E = L["E"]
    th, ph = np.degrees(L["theta"]), np.degrees(L["phi"])
    En = (E - E.min()) / (E.max() - E.min())
    sx, sy, sz = 7.5 / 360, 3.75 / 180, 2.4

    def surf_f(u, v):
      return np.array([u * sx, v * sy, sz * float(interp2(En, th, ph, u, v))])

    surf = ParametricSurface(surf_f, u_range=(-180, 180), v_range=(-90, 90), resolution=(73, 37))
    surf.set_color_by_rgba_func(lambda p: color_to_rgba(heat(p[2] / sz)))
    surf.set_shading(0.25, 0.2, 0.2)
    frame.reorient(-20, 55, 0)
    frame.set_height(8.5).move_to(OUT * 0.9 + DOWN * 0.3)
    axl = VGroup(Line(LEFT * 3.9 + DOWN * 2.0, RIGHT * 3.9 + DOWN * 2.0).set_stroke(GREY_B, 2),
                 Line(LEFT * 3.9 + DOWN * 2.0, LEFT * 3.9 + UP * 2.0).set_stroke(GREY_B, 2))
    tl_ = TT(r"turn about the symmetry axis: $-180^\circ\ \dots\ 180^\circ$", 26, GREY_A).to_edge(DOWN, buff=0.35)
    pl_ = TT(r"turn about a perpendicular axis: $-90^\circ\ \dots\ 90^\circ$", 26, GREY_A).next_to(tl_, UP, 0.12)
    fix(tl_, pl_)
    self.play(ShowCreation(surf), run_time=2.5)
    self.play(FadeIn(tl_), FadeIn(pl_), run_time=0.8)
    frame.add_updater(lambda m, dt: m.increment_theta(0.07 * dt))
    gt_m = Sphere(radius=0.12).set_color(GT_C).move_to(surf_f(0, 0) + OUT * 0.15)
    sy_m = Group(*[Sphere(radius=0.12).set_color(GT_C).move_to(surf_f(a, 0) + OUT * 0.15) for a in (-180, 180)])
    gl1 = TT(r"true pose", 28, GT_C)
    gl2 = TT(r"symmetric copy ($180^\circ$)", 28, GT_C)
    lg = VGroup(gl1, gl2).arrange(DOWN, aligned_edge=LEFT).to_corner(UR, buff=0.5)
    fix(lg)
    self.until(0.45)
    self.play(FadeIn(gt_m, scale=0.3), FadeIn(gl1), run_time=1.0)
    self.play(FadeIn(sy_m, scale=0.3), FadeIn(gl2), run_time=1.0)
    rc = caption(r"real energy of the trained field for this test crop (ground truth used only to draw the slice); colour: low $\to$ high E")
    fix(rc)
    self.play(FadeOut(pl_), tl_.animate.shift(UP * 0.45), FadeIn(rc), run_time=0.8)
    self.until(1.0)

    # ---- beat 11: exact in the pose: a marker moves along the slice, the rendering follows the pose exactly
    d = self.beat(11)
    mk = Sphere(radius=0.1).set_color(POSE_C)
    av = ValueTracker(0)
    mk.add_updater(lambda m: m.move_to(surf_f(av.get_value(), 0) + OUT * 0.15))
    self.add(mk)
    ex = TT(r"$g\,\mathrm{Exp}(\delta\omega)$: every sample point moves by exactly $\delta\omega$ in the object frame", 28, POSE_C)
    ex2 = TT(r"nothing in $\hat{\mathbf z}$ comes from another image", 28, GREY_A)
    exs = VGroup(ex, ex2).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.75)
    fix(exs)
    self.play(FadeOut(rc), FadeOut(tl_), FadeIn(ex), av.animate.set_value(60), run_time=0.35 * d)
    self.play(FadeIn(ex2), av.animate.set_value(0), run_time=0.3 * d)
    self.until(1.0)
    frame.clear_updaters()
    mk.clear_updaters()
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not frame])), run_time=0.8)
    self.finish()


def interp2(Z, xs, ys, x, y):
  """bilinear interpolation of Z[len(ys), len(xs)] at (x, y)"""
  i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
  j = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
  tx = (x - xs[i]) / (xs[i + 1] - xs[i])
  ty = (y - ys[j]) / (ys[j + 1] - ys[j])
  return (Z[j, i] * (1 - tx) * (1 - ty) + Z[j, i + 1] * tx * (1 - ty) + Z[j + 1, i] * (1 - tx) * ty + Z[j + 1, i + 1] * tx * ty)
