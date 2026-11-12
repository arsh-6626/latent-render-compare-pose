from common import *


class S01(L3Scene):
  SCENE = "S01"

  def construct(self):
    tk, ry = load("tokens.npz"), load("rays.npz")
    R_cam = ry["gt_R_cam"]          # egocentric rotation of the GT pose (object -> camera), from the field's geometry code
    S = tk["S"]                     # object 5's discrete symmetry (BOP models_info), 180 deg about the object z axis
    frame = self.frame

    # ---- beat 1: the real crop and the object at its pose
    d = self.beat(1)
    crop = ImageMobject(os.path.join(DATA, "crop.png"), height=4.6).move_to(LEFT * 3.7 + DOWN * 0.1)
    box = SurroundingRectangle(crop, buff=0).set_stroke(GREY_B, 1.5)
    lab = TT(r"T-LESS test crop (object 5)", 26, GREY_A).next_to(box, DOWN, 0.2)
    mesh = mesh5(scale=30)
    mesh.apply_matrix(CV2M @ R_cam, about_point=ORIGIN)
    mesh.move_to(RIGHT * 3.4 + DOWN * 0.1)
    c0 = mesh.get_center()
    fix(crop, box)
    self.play(FadeIn(crop), ShowCreation(box), run_time=1.2)
    fix(lab)
    self.play(FadeIn(lab), FadeIn(mesh, scale=0.9), run_time=1.4)
    # body axes of the object at its pose
    ax_cols = [RED_C, GREEN_C, BLUE_C]
    axes = VGroup(*[Arrow(c0, c0 + 1.7 * (CV2M @ R_cam @ e), buff=0, stroke_width=4, fill_color=c).set_color(c)
                    for e, c in zip(np.eye(3), ax_cols)])
    g_lab = fix(T(r"g = (R,\ t)", 40, POSE_C).move_to(RIGHT * 3.4 + UP * 2.9))
    self.play(GrowArrow(axes[0]), GrowArrow(axes[1]), GrowArrow(axes[2]), Write(g_lab), run_time=0.25 * d)
    six = T(r"3 \text{ rotation} + 3 \text{ translation} = 6 \text{ degrees of freedom}", 28, GREY_A).next_to(g_lab, DOWN, 0.15)
    fix(six)
    self.play(FadeIn(six, UP * 0.2), run_time=1.0)
    # turntable to show it is a 3-D object, then back
    grp = Group(mesh, axes)
    self.play(Rotate(grp, 2 * PI, axis=UP, about_point=c0), run_time=0.35 * d)
    self.until(1.0)

    # ---- beat 2: the symmetry
    d = self.beat(2)
    s_dir = CV2M @ R_cam @ np.array([0, 0, 1.0])  # symmetry axis (object z) in display coordinates
    sym_line = DashedLine(c0 - 2.3 * s_dir, c0 + 2.3 * s_dir, dash_length=0.12).set_stroke(YELLOW, 3)
    ghost = mesh.copy().set_opacity(0.25)
    self.play(ShowCreation(sym_line), FadeOut(axes), FadeOut(six), run_time=1.2)
    self.add(ghost)
    arc_lab = fix(T(r"180^\circ", 34, YELLOW).next_to(mesh, RIGHT, 0.3))
    self.play(Rotate(mesh, PI, axis=s_dir, about_point=c0), FadeIn(arc_lab), run_time=0.3 * d)
    same = fix(TT(r"same image", 32, GT_C).next_to(mesh, DOWN, 0.5))
    self.play(FadeIn(same, UP * 0.2), run_time=0.8)
    # two poses as two points of SO(3) (axis-angle ball), and a continuum for a rotationally symmetric object
    self.play(FadeOut(Group(ghost, mesh, sym_line, arc_lab, same, lab)), crop.animate.scale(0.62).to_corner(UL, buff=0.4),
              box.animate.scale(0.62).to_corner(UL, buff=0.4), g_lab.animate.scale(0.8).to_corner(UR, buff=0.5), run_time=1.5)
    bl = ball().move_to(RIGHT * 1.0 + DOWN * 0.2)
    bc = bl.get_center()
    ball_lab = fix(T(r"\mathrm{SO}(3):\ \text{axis} \times \text{angle} \le \pi", 30, GREY_A).next_to(bl, RIGHT, 0.4).shift(DOWN * 1.6))
    self.play(FadeIn(bl), FadeIn(ball_lab), frame.animate.reorient(-25, -10, 0), run_time=1.6)
    p1 = bc + rot_to_ball(R_cam)
    p2 = bc + rot_to_ball(R_cam @ S)
    d1, d2 = [Sphere(radius=0.1).set_color(GT_C).move_to(p) for p in (p1, p2)]
    self.play(FadeIn(d1, scale=0.3), FadeIn(d2, scale=0.3), run_time=1.0)
    l2 = Line(p1, p2).set_stroke(GT_C, 2, 0.5)
    self.play(ShowCreation(l2), run_time=0.8)
    self.until(0.72)
    orbit = [bc + rot_to_ball(R_cam @ lm.so3_exp(a * np.array([0, 0, 1.0]))) for a in np.linspace(0, 2 * PI, 241)]
    curve = poly3d(np.array(orbit), YELLOW, 3)
    cl = TT(r"if it were rotationally symmetric: a whole curve", 28, YELLOW).to_edge(DOWN, buff=0.25)
    fix(cl)
    self.play(ShowCreation(curve), FadeIn(cl), run_time=min(2.5, 0.25 * d))
    self.until(1.0)

    # ---- beat 3: a distribution on a curved group
    d = self.beat(3)
    dist = T(r"p(g \mid I),\qquad g \in \mathrm{SE}(3)", 52).to_edge(UP, buff=0.5)
    fix(dist)
    self.play(Write(dist), FadeOut(cl), run_time=1.5)
    frame.add_updater(lambda m, dt: m.increment_theta(0.12 * dt))
    self.play(Indicate(d1, scale_factor=1.6, color=WHITE), Indicate(d2, scale_factor=1.6, color=WHITE), run_time=1.6)
    self.until(1.0)
    frame.clear_updaters()
    self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
    self.finish()
