from common import *


class S02(L3Scene):
  SCENE = "S02"

  def construct(self):
    tk, ry = load("tokens.npz"), load("rays.npz")
    R_cam, t_cam, k = ry["gt_R_cam"], ry["gt_t_cam"], tk["k"]
    frame = self.frame

    # ---- beat 1: g = (R, t), 4x4 matrices, composition
    d = self.beat(1)
    cam_o = LEFT * 5.0 + DOWN * 1.6
    cam_ax = VGroup(*[Arrow(cam_o, cam_o + 1.1 * (CV2M @ e), buff=0, stroke_width=4).set_color(c)
                      for e, c in zip(np.eye(3), [RED_C, GREEN_C, BLUE_C])])
    cam_lab = TT("camera", 26, GREY_A).next_to(cam_o, LEFT, 0.3)
    obj_o = RIGHT * 0.2 + UP * 0.6
    mesh = mesh5(scale=18).apply_matrix(CV2M @ R_cam, about_point=ORIGIN).move_to(obj_o)
    obj_ax = VGroup(*[Arrow(obj_o, obj_o + 1.3 * (CV2M @ R_cam @ e), buff=0, stroke_width=4).set_color(c)
                      for e, c in zip(np.eye(3), [RED_C, GREEN_C, BLUE_C])])
    t_arrow = Arrow(cam_o, obj_o, buff=0, stroke_width=5).set_color(POSE_C)
    t_lab = T("t", 40, POSE_C).next_to(t_arrow.get_center(), UL, 0.1)
    R_lab = T("R", 40, POSE_C).next_to(obj_ax, RIGHT, 0.2).shift(UP * 0.6)
    self.play(LaggedStart(*[GrowArrow(a) for a in cam_ax], lag_ratio=0.2), FadeIn(cam_lab), run_time=1.2)
    self.play(GrowArrow(t_arrow), FadeIn(t_lab), run_time=1.2)
    self.play(FadeIn(mesh), LaggedStart(*[GrowArrow(a) for a in obj_ax], lag_ratio=0.2), FadeIn(R_lab), run_time=1.5)
    mat = T(r"g=\begin{pmatrix} R & t \\ 0^\top & 1 \end{pmatrix}\in \mathrm{SE}(3)", 44).to_corner(UR, buff=0.6)
    comp = T(r"g_1 g_2=\begin{pmatrix} R_1R_2 & R_1t_2+t_1 \\ 0^\top & 1 \end{pmatrix}", 40).next_to(mat, DOWN, 0.5)
    self.play(Write(mat), run_time=1.6)
    self.play(FadeIn(comp, DOWN * 0.2), run_time=1.4)
    self.until(1.0)

    # ---- beat 2: crop-relative parametrisation (real crop camera of this test crop)
    d = self.beat(2)
    self.play(FadeOut(Group(cam_ax, cam_lab, mesh, obj_ax, t_arrow, t_lab, R_lab, comp)), mat.animate.scale(0.8).to_corner(UR, buff=0.4),
              run_time=1.0)
    s, x0, y0, bw, bh, fr, fx, fy, cx, cy = [float(v) for v in k]
    W, H = 720.0, 540.0  # T-LESS Primesense images
    sc = 6.2 / W
    img_c = LEFT * 3.4 + UP * 0.2
    to_scr = lambda u, v: img_c + np.array([(u - W / 2) * sc, (H / 2 - v) * sc, 0])
    plane = Rectangle(W * sc, H * sc).move_to(img_c).set_stroke(GREY_B, 2).set_fill(BLACK, 0.4)
    plane_lab = TT(r"image plane (720 $\times$ 540 px)", 24, GREY_B).next_to(plane, UP, 0.1).align_to(plane, LEFT)
    side = 224.0 / s  # crop side in image pixels
    crop_img = ImageMobject(os.path.join(DATA, "crop.png"), height=side * sc).move_to(to_scr(x0 + side / 2, y0 + side / 2))
    crop_box = SurroundingRectangle(crop_img, buff=0).set_stroke(OBS_C, 2.5)
    self.play(ShowCreation(plane), FadeIn(plane_lab), run_time=1.0)
    self.play(FadeIn(crop_img), ShowCreation(crop_box), run_time=1.0)
    # magnified inset of the crop (exact geometry of this crop camera)
    M = 3.6 / side  # inset scale (units per image px)
    ins_c = RIGHT * 2.6 + UP * 0.9
    ins = ImageMobject(os.path.join(DATA, "crop.png"), height=3.6).move_to(ins_c)
    ins_box = SurroundingRectangle(ins, buff=0).set_stroke(OBS_C, 2.5)
    links = VGroup(Line(crop_box.get_corner(UR), ins_box.get_corner(UL)), Line(crop_box.get_corner(DR), ins_box.get_corner(DL)))
    links.set_stroke(OBS_C, 1, 0.5)
    self.play(FadeIn(ins), ShowCreation(ins_box), ShowCreation(links), run_time=1.4)
    to_ins = lambda u, v: ins_c + np.array([(u - (x0 + side / 2)) * M, ((y0 + side / 2) - v) * M, 0])
    bc = to_ins(x0 + side / 2, y0 + side / 2)  # crop centre = box centre (LiePose's dynamic zoom-in crop)
    uc = to_ins(fx * t_cam[0] / t_cam[2] + cx, fy * t_cam[1] / t_cam[2] + cy)  # projected object centre
    dbc = Dot(bc, radius=0.07).set_color(OBS_C)
    duc = Dot(uc, radius=0.07).set_color(POSE_C)
    corner = np.array([uc[0], bc[1], 0])
    dx = Line(bc, corner).set_stroke(POSE_C, 4)
    dy = Line(corner, uc).set_stroke(POSE_C, 4)
    dxl = T(r"\delta_x", 30, POSE_C).next_to(dx, UP, 0.08)
    dyl = T(r"\delta_y", 30, POSE_C).next_to(dy, RIGHT, 0.08)
    self.play(FadeIn(dbc), FadeIn(duc), run_time=0.8)
    self.play(ShowCreation(dx), ShowCreation(dy), FadeIn(dxl), FadeIn(dyl), run_time=1.0)
    form = T(r"t=\Big(\tfrac{\delta_x}{w},\ \tfrac{\delta_y}{h},\ \tfrac{z}{r}\Big)", 46, POSE_C)
    sc_lab = TT(r"$(w,h)$ box size, $r$ zoom ratio of the crop", 26, GREY_B)
    allo = TT(r"$R$: allocentric (relative to the viewing ray)", 30, GREY_A)
    col = VGroup(form, sc_lab, allo).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(ins_box, DOWN, 0.35).align_to(ins_box, LEFT)
    self.play(Write(form), run_time=1.5)
    self.play(FadeIn(sc_lab), run_time=0.8)
    self.until(0.55)
    self.play(FadeIn(allo, UP * 0.2), run_time=1.0)
    self.until(1.0)

    # ---- beat 3: Lie algebra, Exp on SO(3): left schematic manifold, right the actual rotation of the object
    d = self.beat(3)
    self.play(FadeOut(Group(plane, plane_lab, crop_img, crop_box, ins, ins_box, links, dbc, duc, dx, dy, dxl, dyl, col, mat)), run_time=0.8)
    frame.reorient(0, 0, 0)
    man_c = LEFT * 3.4 + DOWN * 0.6
    man = Sphere(radius=2.0).move_to(man_c).set_color(BLUE_E).set_opacity(0.55)
    man.set_shading(0.4, 0.2, 0.3)
    e_pt = man_c + 2.0 * UP
    tplane = Square(2.6).set_fill(TEAL_E, 0.35).set_stroke(TEAL_B, 1).rotate(PI / 2, RIGHT).move_to(e_pt)
    tp_lab = T(r"\mathfrak{se}(3) \cong \mathbb{R}^6", 34, TEAL_B).next_to(tplane, UP, 0.15).shift(LEFT * 0.6)
    e_dot = Dot(e_pt).set_color(WHITE)
    e_lab = T("e", 30).next_to(e_dot, LEFT, 0.12)
    sch = TT("schematic", 22, GREY_B).next_to(man, DOWN, 0.15)
    rod = T(r"\mathrm{Exp}(\omega)=I+\frac{\sin\theta}{\theta}[\omega]_\times+\frac{1-\cos\theta}{\theta^2}[\omega]_\times^2,\quad \theta=\|\omega\|",
            34).to_edge(UP, buff=0.35)
    fix(rod)
    frame.reorient(0, -18, 0)
    self.play(FadeIn(man), FadeIn(tplane), FadeIn(e_dot), FadeIn(e_lab), FadeIn(tp_lab), FadeIn(sch), run_time=1.4)
    # the object on the right turns about axis omega by |omega|
    obj_c = RIGHT * 3.4 + DOWN * 0.5
    mesh2 = mesh5(scale=22).apply_matrix(CV2M @ R_cam, about_point=ORIGIN).move_to(obj_c)
    axis = normalize(np.array([0.5, 0.8, 0.35]))
    ax_line = DashedLine(obj_c - 2.2 * axis, obj_c + 2.2 * axis).set_stroke(NOISE_C, 3)
    self.play(FadeIn(mesh2), ShowCreation(ax_line), Write(rod), run_time=1.6)
    theta = ValueTracker(0.0)
    tdir = normalize(np.array([1.0, 0, -0.35]))  # direction in the tangent plane (schematic)

    def vec_upd(m):
      L = theta.get_value() * 0.9
      m.put_start_and_end_on(e_pt, e_pt + L * tdir + 1e-4 * RIGHT)

    vec = Arrow(e_pt, e_pt + 0.01 * tdir, buff=0, stroke_width=5).set_color(NOISE_C)
    vec.add_updater(vec_upd)
    om_lab = T(r"\omega", 36, NOISE_C).add_updater(lambda m: m.next_to(vec.get_end(), UP, 0.1))

    def arc_pts():
      th = theta.get_value()
      a = np.linspace(0, th * 0.9 / 2.0, 40)  # geodesic of the same length on the schematic sphere
      return [man_c + 2.0 * (np.cos(x) * UP + np.sin(x) * tdir) for x in a]

    arc = VMobject().set_stroke(YELLOW, 4)
    arc.add_updater(lambda m: m.set_points_as_corners(arc_pts()) if theta.get_value() > 0.02 else m.set_points_as_corners([e_pt, e_pt]))
    ex_lab = T(r"\mathrm{Exp}(\omega)", 32, YELLOW).add_updater(lambda m: m.next_to(arc_pts()[-1], RIGHT, 0.1))
    self.add(vec, om_lab, arc, ex_lab)
    prev = [0.0]

    def rot_upd(m):
      th = theta.get_value()
      m.rotate(th - prev[0], axis=axis, about_point=obj_c)
      prev[0] = th

    mesh2.add_updater(rot_upd)
    ang = DecimalNumber(0, num_decimal_places=0, font_size=36).set_color(YELLOW)
    ang.add_updater(lambda m: m.set_value(np.degrees(theta.get_value())))
    eqp = T(r"\theta=\|\omega\| =", 36, YELLOW).move_to(RIGHT * 2.9 + DOWN * 2.6)
    deg = T(r"^\circ", 36, YELLOW)
    ang.add_updater(lambda m: m.next_to(eqp, RIGHT, 0.15))
    deg.add_updater(lambda m: m.next_to(ang, RIGHT, 0.05).align_to(ang, UP))
    ang_lab = VGroup(eqp, ang, deg)
    fix(ang_lab)
    self.add(ang_lab)
    self.play(theta.animate.set_value(2.4), run_time=0.45 * d, rate_func=smooth)
    self.until(1.0)

    # ---- beat 4: Log and the local parametrisation g Exp(xi)
    d = self.beat(4)
    self.play(theta.animate.set_value(0.8), run_time=0.3 * d)
    loc = T(r"g' = g\,\mathrm{Exp}(\xi),\quad \xi=\begin{pmatrix} v\\ \omega\end{pmatrix}\in\mathbb{R}^6,\quad "
            r"\mathrm{Exp}(\xi)=\big(\mathrm{Exp}(\omega),\ V(\omega)\,v\big)", 34).to_edge(DOWN, buff=0.45)
    lg = T(r"\xi=\mathrm{Log}(g^{-1}g')", 34, TEAL_B).next_to(loc, UP, 0.25)
    fix(loc, lg)
    self.play(Write(loc), FadeOut(ang_lab), run_time=1.8)
    self.play(FadeIn(lg, UP * 0.2), run_time=0.8)
    self.until(1.0)
    for m in (vec, om_lab, arc, ex_lab, mesh2, ang, deg):
      m.clear_updaters()
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
