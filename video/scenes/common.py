"""Shared pieces of the LaRC explainer: colours, narration timing, real data, mesh mobject, token grids."""
import json
import os
import sys

import numpy as np
from manimlib import *  # noqa: F401,F403

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import larc_math as lm  # noqa: E402

DATA = "/media/kavinder/hdd3/larc_video/data"
AUD = "/media/kavinder/hdd3/larc_video/audio"
TIMING = "/media/kavinder/hdd3/larc_video/render/timing"

# ------------------------------------------------------------------ colours (fixed meaning for the whole video)
BG = "#0D1117"
OBS_C = "#4FC3F7"     # observation z
REN_C = "#FFB74D"     # world-model rendering z_hat (and LaRC)
DIFF_C = "#F06292"    # difference z_hat - z
LAT_C = "#BA68C8"     # field latent h
POSE_C = "#FFEE58"    # hypothesis g_t
GT_C = "#81C784"      # ground truth / data pose g_0
CTRL_C = "#90A4AE"    # control (no world model)
LIE_C = "#7986CB"     # LiePose
FLOW_C = "#4DB6AC"    # flow matching
NOISE_C = "#E57373"   # noise epsilon
DIM_C = "#5C6370"
TXT_C = "#E6E6E6"

# ------------------------------------------------------------------ narration timing
DUR = json.load(open(os.path.join(AUD, "durations.json"))) if os.path.exists(os.path.join(AUD, "durations.json")) else {}


class NarratedMixin:
  """beat(k) starts narration beat k of this scene: waits until the previous beat's audio has finished (plus a gap),
  records the actual start time (used to place the audio), and returns the beat's duration."""
  SCENE = "S00"
  GAP = 0.25

  def setup(self):
    super().setup()
    self.beat_log = {}
    self.next_free = 0.4

  def beat(self, k, gap=None):
    key = f"{self.SCENE}.{k}"
    start = self.next_free + (self.GAP if gap is None else gap) if self.beat_log else self.next_free
    if start > self.time:
      self.wait(start - self.time)
    self.beat_log[key] = float(self.time)
    d = DUR.get(key, 6.0)
    self.next_free = self.time + d
    return d

  def until(self, frac_of_beat_end=1.0, key=None):
    """wait until the current beat's audio has ended (or a fraction into it)"""
    if not self.beat_log:
      return
    k = key or list(self.beat_log)[-1]
    t = self.beat_log[k] + DUR.get(k, 6.0) * frac_of_beat_end
    if t > self.time:
      self.wait(t - self.time)

  def finish(self, tail=0.6):
    end = self.next_free + tail
    if end > self.time:
      self.wait(end - self.time)
    os.makedirs(TIMING, exist_ok=True)
    json.dump({"beats": self.beat_log, "total": float(self.time)}, open(os.path.join(TIMING, f"{self.SCENE}.json"), "w"), indent=1)


class LScene(NarratedMixin, Scene):
  default_camera_config = dict(background_color=BG)


class L3Scene(NarratedMixin, Scene):
  """3-D content with a flat default camera (the frame is reoriented explicitly where needed)"""
  default_camera_config = dict(background_color=BG)


# ------------------------------------------------------------------ text helpers
def T(s, size=36, color=TXT_C, **kw):
  return Tex(s, font_size=size, **kw).set_color(color) if "t2c" not in kw else Tex(s, font_size=size, **kw)


def TT(s, size=30, color=TXT_C, **kw):
  return TexText(s, font_size=size, **kw).set_color(color)


def caption(s, size=24):
  """small grey caption at the bottom edge (provenance / honesty labels)"""
  return TexText(s, font_size=size).set_color("#9AA4B2").to_edge(DOWN, buff=0.18)


def fix(*mobs):
  for m in mobs:
    m.fix_in_frame()
  return mobs[0] if len(mobs) == 1 else mobs


# ------------------------------------------------------------------ real data
def load(name):
  return np.load(os.path.join(DATA, name), allow_pickle=True)


def token_pca():
  """one PCA basis on observed + rendered tokens (as tools/real_data.py) -> colour function"""
  t = load("tokens.npz")
  X = np.concatenate([t["z"], t["zh_gt"], t["zh_wr"]], 0)
  mu = X.mean(0)
  _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
  P = Vt[:3]
  allp = (X - mu) @ P.T
  lo, hi = np.percentile(allp, 2, 0), np.percentile(allp, 98, 0)
  return lambda a: np.clip(((np.asarray(a, np.float64) - mu) @ P.T - lo) / (hi - lo), 0, 1)


def rgb_hex(c):
  c = np.clip(np.asarray(c), 0, 1)
  return "#%02x%02x%02x" % tuple(int(round(255 * x)) for x in c[:3])


def heat(v):
  """0..1 -> dark blue .. magenta .. yellow (perceptual-ish)"""
  v = float(np.clip(v, 0, 1))
  stops = [(0.0, (0.05, 0.07, 0.20)), (0.35, (0.45, 0.10, 0.55)), (0.7, (0.94, 0.38, 0.45)), (1.0, (1.0, 0.92, 0.45))]
  for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
    if v <= b:
      s = (v - a) / (b - a)
      return rgb_hex(np.array(ca) * (1 - s) + np.array(cb) * s)
  return rgb_hex(stops[-1][1])


class TokenGrid(VGroup):
  """n x n grid of squares, filled with given colours (row-major, row 0 at the top as in the image)"""

  def __init__(self, colors, n=8, side=0.36, gap=0.03, stroke=0.6, **kw):
    super().__init__(**kw)
    for i in range(n):
      for j in range(n):
        sq = Square(side).set_fill(colors[i * n + j], 1).set_stroke(BG, stroke)
        sq.move_to(np.array([(j - (n - 1) / 2) * (side + gap), ((n - 1) / 2 - i) * (side + gap), 0]))
        self.add(sq)

  def recolor(self, colors):
    for sq, c in zip(self, colors):
      sq.set_fill(c, 1)
    return self


# ------------------------------------------------------------------ CAD mesh as a lit manimgl surface (flat shading)
class MeshMob(Surface):
  def __init__(self, V, F, color=GREY_B, scale=1.0, **kw):
    self._V = np.asarray(V, np.float64) * scale
    self._F = np.asarray(F, np.int64)
    super().__init__(color=color, resolution=(2, 2), **kw)

  def init_points(self):
    tri = self._V[self._F]  # (nf, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    n = np.cross(e1, tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(n, e1)
    P = tri.reshape(-1, 3)
    eps = 1e-3
    self.set_points(P)
    self.data["du_point"][:] = P + eps * np.repeat(e1, 3, 0)
    self.data["dv_point"][:] = P + eps * np.repeat(e2, 3, 0)

  def compute_triangle_indices(self):
    self.triangle_indices = np.arange(len(self._F) * 3)
    return self.triangle_indices


CV2M = np.diag([1.0, -1.0, -1.0])  # OpenCV camera frame (x right, y down, z forward) -> manim (x right, y up, z to viewer)


def mesh5(scale=30.0, color="#C9CED6", **kw):
  m = load("mesh5.npz")
  return MeshMob(m["v"], m["f"], color=color, scale=scale, **kw)


def apply_rot(mob, R, about=ORIGIN):
  """rotate a mobject by the 3x3 matrix R about a point (manim coordinates)"""
  mob.apply_matrix(R, about_point=about)
  return mob


# ------------------------------------------------------------------ axis-angle ball for SO(3)
BALL_R = 2.6  # display radius of the ball (angle pi)


def rot_to_ball(R):
  w = lm.so3_log(R)
  return w / np.pi * BALL_R


def ball(radius=BALL_R, opacity=0.08):
  s = Sphere(radius=radius, resolution=(41, 21)).set_color(BLUE_E).set_opacity(opacity)
  s.deactivate_depth_test()
  eq = Circle(radius=radius).set_stroke(BLUE_D, 1.2, 0.5)
  mer = Circle(radius=radius).rotate(PI / 2, RIGHT).set_stroke(BLUE_D, 1.2, 0.35)
  return Group(s, eq, mer)


def poly3d(points, color, width=2.0, opacity=1.0, jump=0.6):
  """polyline through 3-D points, broken where it jumps across the ball (antipodal identification at angle pi)"""
  g = VGroup()
  cur = [points[0]]
  for a, b in zip(points[:-1], points[1:]):
    if np.linalg.norm(b - a) > jump * BALL_R:
      if len(cur) > 1:
        g.add(VMobject().set_points_as_corners(cur))
      cur = [b]
    else:
      cur.append(b)
  if len(cur) > 1:
    g.add(VMobject().set_points_as_corners(cur))
  return g.set_stroke(color, width, opacity)


class PathGrower:
  """trajectories (list of (n_steps, 3) arrays) drawn up to a ValueTracker step, with moving heads; polylines are rebuilt
  every frame and broken at antipodal jumps of the axis-angle ball"""

  def __init__(self, paths, color, width=2.0, opacity=0.85, head_r=0.075, head_color=None):
    self.paths, self.color, self.width, self.opacity = paths, color, width, opacity
    self.prog = ValueTracker(0)
    self.lines = VGroup()
    self.heads = DotCloud([p[0] for p in paths], radius=head_r).set_color(head_color or color)
    self.lines.add_updater(self._upd_lines)
    self.heads.add_updater(lambda m: m.set_points([p[self._i()] for p in self.paths]))

  def _i(self):
    return int(np.clip(round(self.prog.get_value()), 0, len(self.paths[0]) - 1))

  def _upd_lines(self, m):
    i = self._i()
    new = VGroup(*[poly3d(p[:i + 1], self.color, self.width, self.opacity) for p in self.paths]) if i > 0 else VGroup()
    m.set_submobjects(list(new))

  def stop(self):
    self.lines.clear_updaters()
    self.heads.clear_updaters()


def small_axes(x_range, y_range, width, height, y_labels, x_labels, font=24):
  ax = Axes(x_range=x_range, y_range=y_range, width=width, height=height,
            axis_config=dict(stroke_color=GREY_B, include_tip=False))
  labs = VGroup()
  for v in x_labels:
    labs.add(T(f"{v:g}", font, GREY_B).next_to(ax.c2p(v, y_range[0]), DOWN, 0.12))
  for v in y_labels:
    labs.add(T(f"{v:g}", font, GREY_B).next_to(ax.c2p(x_range[0], v), LEFT, 0.12))
  ax.add(labs)
  return ax


def interp2(Z, xs, ys, x, y):
  """bilinear interpolation of Z[len(ys), len(xs)] at (x, y)"""
  i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
  j = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
  tx = (x - xs[i]) / (xs[i + 1] - xs[i])
  ty = (y - ys[j]) / (ys[j + 1] - ys[j])
  return (Z[j, i] * (1 - tx) * (1 - ty) + Z[j, i + 1] * tx * (1 - ty) + Z[j + 1, i] * (1 - tx) * ty + Z[j + 1, i + 1] * tx * ty)


def landscape_surface(sx=7.5 / 360, sy=3.75 / 180, sz=2.4):
  """the real energy landscape (tools/real_data.py) as a 3-D surface; returns (surface, point function)"""
  L = load("landscape.npz")
  E = L["E"]
  th, ph = np.degrees(L["theta"]), np.degrees(L["phi"])
  En = (E - E.min()) / (E.max() - E.min())
  f = lambda u, v: np.array([u * sx, v * sy, sz * float(interp2(En, th, ph, u, v))])
  surf = ParametricSurface(f, u_range=(-180, 180), v_range=(-90, 90), resolution=(73, 37))
  surf.set_color_by_rgba_func(lambda p: color_to_rgba(heat(p[2] / sz)))
  surf.set_shading(0.25, 0.2, 0.2)
  return surf, f
