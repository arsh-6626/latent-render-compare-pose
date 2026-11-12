from common import *

BOX = 2.0


class S08(L3Scene):
  SCENE = "S08"

  def construct(self):
    dn, tv = load("density.npz"), load("train_views.npz")
    r_o = float(dn["r_o"])
    frame = self.frame
    frame.reorient(25, 65, 0)
    frame.set_height(10)

    # ---- beat 1: posed training crops; loss = energy at the ground-truth pose
    d = self.beat(1)
    cube = Cube(side_length=2 * BOX).set_opacity(0).set_stroke(GREY_A, 1.2, 0.7) if False else None
    edges = VGroup(*[Line(BOX * np.array(a), BOX * np.array(b)) for a, b in
                     [((-1, -1, -1), (1, -1, -1)), ((-1, 1, -1), (1, 1, -1)), ((-1, -1, 1), (1, -1, 1)), ((-1, 1, 1), (1, 1, 1)),
                      ((-1, -1, -1), (-1, 1, -1)), ((1, -1, -1), (1, 1, -1)), ((-1, -1, 1), (-1, 1, 1)), ((1, -1, 1), (1, 1, 1)),
                      ((-1, -1, -1), (-1, -1, 1)), ((1, -1, -1), (1, -1, 1)), ((-1, 1, -1), (-1, 1, 1)), ((1, 1, -1), (1, 1, 1))]])
    edges.set_stroke(GREY_A, 1.2, 0.7)
    views = DotCloud(4.2 * tv["view"], radius=0.035).set_color(OBS_C).set_opacity(0.8)
    self.play(ShowCreation(edges), run_time=1.0)
    self.play(FadeIn(views, lag_ratio=0.002), run_time=2.0)
    vl = TT(r"viewing directions of the training crops of object 5 (real)", 26, OBS_C).to_edge(DOWN, buff=0.35)
    fix(vl)
    loss = T(r"\min_\phi\ \mathbb{E}_{(\mathbf z,\,g^\star)}\ E(g^\star;\mathbf z)", 46).to_corner(UL, buff=0.45)
    nl = TT(r"180,224 training crops (T-LESS), each at its ground-truth pose $g^\star$", 26, GREY_A).next_to(loss, DOWN, 0.2).align_to(loss, LEFT)
    fix(loss, nl)
    self.play(FadeIn(vl), Write(loss), run_time=1.6)
    self.play(FadeIn(nl), run_time=0.8)
    frame.add_updater(lambda m, dt: m.increment_theta(0.1 * dt))
    self.until(1.0)

    # ---- beat 2: the density that emerged (never supervised) next to the CAD model it never saw
    d = self.beat(2)
    no = TT(r"reconstruction only: no contrastive term, never a wrong pose,\\ density never supervised", 26, GREY_A)
    no.next_to(nl, DOWN, 0.15).align_to(nl, LEFT)
    fix(no)
    self.play(FadeIn(no), FadeOut(views), FadeOut(vl), run_time=1.0)
    G, s = dn["grid"], dn["sigma"]
    keep = s > np.percentile(s, 88)
    P, hc = BOX * G[keep], dn["h_pca"][keep]
    hc = (hc - np.percentile(hc, 2, 0)) / (np.percentile(hc, 98, 0) - np.percentile(hc, 2, 0))
    cloud = DotCloud(P, radius=0.05)
    cloud.set_rgba_array(np.concatenate([np.clip(0.25 + 0.75 * hc, 0, 1), 0.85 * np.ones((len(P), 1))], 1))
    dl = TT(r"learned density $\sigma(x)$ (highest 12\% of a $40^3$ grid), coloured by the latent $\mathbf h(x)$", 26, LAT_C)
    dl.to_edge(DOWN, buff=0.35)
    fix(dl)
    self.play(FadeIn(cloud, lag_ratio=0.001), FadeIn(dl), run_time=0.3 * d)
    mesh = mesh5(scale=BOX / r_o).set_opacity(0.0)
    mesh.shift(RIGHT * 0.0)
    self.wait(0.5)
    cmp_ = TT(r"CAD model (never seen by the field)", 26, GREY_A).to_corner(UR, buff=0.5).shift(DOWN * 0.8)
    fix(cmp_)
    self.add(mesh)
    self.play(mesh.animate.set_opacity(0.45), FadeIn(cmp_), run_time=1.6)
    self.wait(0.15 * d)
    self.play(mesh.animate.set_opacity(0.0), run_time=1.2)
    self.until(1.0)

    # ---- beat 3: fitted once, then frozen
    d = self.beat(3)
    fr = T(r"\phi:\ \text{fitted once (100k steps), then frozen}", 40, REN_C).to_edge(DOWN, buff=1.0)
    fix(fr)
    self.play(FadeIn(fr, UP * 0.2), run_time=1.0)
    self.until(1.0)
    frame.clear_updaters()
    self.play(FadeOut(Group(*[m for m in self.mobjects if m is not frame])), run_time=0.8)
    self.finish()
