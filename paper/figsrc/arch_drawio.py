"""Architecture diagrams as draw.io files (plain mxGraph XML; open in draw.io / diagrams.net, File > Open from device),
with thumbnails from real inference (arch_thumbs.py) embedded as base64, so each .drawio file is self-contained.
Run: python arch_thumbs.py; python arch_drawio.py -> LARp/paper/fig/arch/*.drawio
Content follows the method section (Sec. 3) and the code (tless_field_dit.py, sp_field_dit.py, tless_field.py)."""
import base64
import html
import os

OUT = "/home/kavinder/ARNABI_ARSH/LARp/paper/fig/arch"
os.makedirs(OUT, exist_ok=True)
FONT = "fontFamily=Times New Roman;fontSize=13;"
COL = {"obs": "#dae8fc;strokeColor=#6c8ebf", "wm": "#d5e8d4;strokeColor=#82b366", "den": "#e1d5e7;strokeColor=#9673a6",
       "tok": "#fff2cc;strokeColor=#d6b656", "loss": "#f8cecc;strokeColor=#b85450", "plain": "#f5f5f5;strokeColor=#666666",
       "white": "#ffffff;strokeColor=#666666"}


class D:
  def __init__(self, name):
    self.name, self.cells, self.n = name, [], 2

  def _id(self):
    self.n += 1
    return f"c{self.n}"

  def _cell(self, i, value, st, x, y, w, h):
    self.cells.append(f'<mxCell id="{i}" value="{html.escape(value)}" style="{st}" vertex="1" parent="1">'
                      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

  def box(self, x, y, w, h, text, kind="plain", rounded=True, extra=""):
    st = f"rounded={int(rounded)};whiteSpace=wrap;html=1;fillColor={COL[kind]};{FONT}arcSize=12;{extra}"
    return self._cell(self._id(), text, st, x, y, w, h)

  def group(self, x, y, w, h, title, kind="white"):
    return self.box(x, y, w, h, title, kind, extra="dashed=1;verticalAlign=top;fontStyle=2;opacity=60;")

  def text(self, x, y, w, h, text, size=12, extra=""):
    st = f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;fontFamily=Times New Roman;fontSize={size};{extra}"
    return self._cell(self._id(), text, st, x, y, w, h)

  def img(self, x, y, w, h, name, label=""):
    """a real inference thumbnail (arch_thumbs.py), embedded as base64"""
    b64 = base64.b64encode(open(f"{OUT}/thumbs/{name}.png", "rb").read()).decode()
    st = (f"shape=image;html=1;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;"
          f"fontFamily=Times New Roman;fontSize=11;imageBorder=#999999;image=data:image/png,{b64};")
    return self._cell(self._id(), label, st, x, y, w, h)

  def edge(self, a, b, label="", dashed=False, color="#333333", pts=None, extra=""):
    i = self._id()
    st = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeColor={color};strokeWidth=1.5;"
          f"{FONT}fontSize=11;labelBackgroundColor=#ffffff;" + ("dashed=1;" if dashed else "") + extra)
    geo = '<mxGeometry relative="1" as="geometry">'
    if pts:
      geo += '<Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + "</Array>"
    geo += "</mxGeometry>"
    self.cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{st}" edge="1" parent="1" source="{a}" '
                      f'target="{b}">{geo}</mxCell>')
    return i

  def save(self):
    body = "".join(self.cells)
    xml = (f'<mxfile host="drawio"><diagram name="{self.name}" id="{self.name}"><mxGraphModel dx="1200" dy="800" grid="1" '
           f'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" '
           f'pageHeight="760" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
           f'</mxGraphModel></diagram></mxfile>')
    open(f"{OUT}/{self.name}.drawio", "w").write(xml)
    print("saved", f"{OUT}/{self.name}.drawio")


# ------------------------------------------------------------------ 1. overview: latent render-and-compare inside the sampler
# images: T-LESS object 23 (fig_in_loop's crop), LaRC chain 0 at steps 1 / 61 / 100 (rotation error 123, 24, 2 deg)
d = D("arch_overview")
crop = d.img(20, 240, 110, 110, "crop", "crop I")
enc = d.box(170, 190, 140, 55, "encoder ψ<br>(DINOv2-S, trained)", "obs")
ema = d.box(170, 330, 140, 55, "EMA encoder ψ̄", "obs", extra="dashed=1;")
d.edge(crop, enc); d.edge(crop, ema)
uext = d.box(350, 197, 100, 40, "u", "tok")
zobs = d.img(360, 318, 80, 80, "tokens_obs", "z̄ (observed tokens)")
d.edge(enc, uext); d.edge(ema, zobs)
wm = d.group(490, 20, 610, 345, "world model: rendered at the current hypothesis g_t", "wm")
fld = d.box(510, 60, 170, 60, "feature field F_o<br>(per object)", "wm")
app = d.box(510, 140, 170, 45, "appearance a(z̄)", "wm")
d.edge(zobs, app, pts=[(595, 358)])
d.edge(app, fld)
for j, (s, lab) in enumerate(((1, "step 1"), (61, "step 61"), (100, "step 100"))):
  x = 705 + 90 * j
  d.text(x, 42, 80, 20, lab, 11)
  d.img(x, 65, 80, 80, f"hyp_{s}")
  d.img(x, 160, 80, 80, f"render_{s}")
  d.img(x, 255, 80, 80, f"residual_{s}")
d.text(975, 65, 115, 80, "hypothesis g_t<br>(orange) vs truth<br>(dashed)", 11)
d.text(975, 160, 115, 80, "render ẑ(g_t)<br>(PCA of features)", 11)
d.text(975, 255, 115, 80, "residual<br>‖ẑ − z̄‖² per token", 11)
tk = d.box(1140, 190, 200, 80, "tokens [z̄, ẑ, ẑ−z̄, w, h] + u<br>(64 patches)", "tok")
d.edge(wm, tk, label="ẑ, o, h (stop-grad)", dashed=True)
d.edge(zobs, tk, pts=[(470, 382), (1200, 382)])
d.edge(uext, tk, pts=[(400, 8), (1290, 8)])
den = d.box(1140, 430, 200, 70, "denoiser<br>(4 adaLN DiT blocks; cond. t, r_t)", "den")
d.edge(tk, den)
step = d.box(810, 430, 230, 70, "reverse step on SE(3)<br>g_t → g_{t−1}", "den")
d.edge(den, step, label="ε̂")
gt = d.box(530, 440, 140, 50, "hypothesis g_t", "plain")
d.edge(step, gt)
d.edge(gt, fld, label="render at g_t", pts=[(495, 465), (495, 90)])
d.text(530, 505, 510, 25, "repeated for 100 steps, 16 chains in parallel", 12, "fontStyle=2;")
sel = d.box(1390, 430, 170, 70, "select one of 16<br>(consensus or lowest energy)", "loss")
d.edge(den, sel, label="after 100 steps", dashed=True)
fin = d.img(1420, 540, 110, 110, "final_16", "16 final hypotheses (blue),<br>truth dashed")
d.edge(sel, fin)
d.save()

# ------------------------------------------------------------------ 2. the feature field
# images: the joint field's own tri-planes for object 23 (PCA of the 32 channels), and its render and opacity at g*
d = D("arch_field")
for j, k in enumerate(("plane_xy", "plane_xz", "plane_yz")):
  d.img(195 + 62 * j, 20, 56, 56, k, k[-2:])
x = d.box(20, 240, 130, 60, "point x on a ray<br>(object frame)", "plain")
tri = d.box(200, 140, 170, 80, "tri-planes of object o<br>3 × 128² × 32<br>(bilinear, summed)", "wm")
four = d.box(200, 320, 170, 60, "Fourier features of x<br>(4 octaves)", "wm")
d.edge(x, tri, pts=[(85, 180)]); d.edge(x, four, pts=[(85, 350)])
mlp = d.box(420, 230, 150, 80, "MLP 2 × 128, GELU<br>→ latent h(x)", "wm")
d.edge(tri, mlp, pts=[(495, 180)]); d.edge(four, mlp, pts=[(495, 350)])
dens = d.box(620, 140, 190, 60, "density σ = softplus(ℓ(h) − 1)<br>(0 outside the box)", "wm")
head_ = d.box(620, 320, 190, 60, "head 2 × 256 (FiLM by a)<br>→ feature f_obj(x)", "wm")
d.edge(mlp, dens, pts=[(715, 270)]); d.edge(mlp, head_, pts=[(715, 270)])
zt = d.img(450, 420, 70, 70, "tokens_obs", "z̄")
apc = d.box(620, 425, 190, 60, "appearance a = GELU(Dense(μ, σ of z̄))", "obs")
d.edge(zt, apc); d.edge(apc, head_)
vr = d.box(870, 220, 200, 100, "volume rendering<br>16 × 16 rays, 32 samples<br>f = o·f_obj + (1 − o)·f_bg", "wm")
d.edge(dens, vr, pts=[(970, 170)]); d.edge(head_, vr, pts=[(970, 350)])
bg = d.box(870, 425, 200, 60, "background f_bg = bg(a)", "obs")
d.edge(apc, bg); d.edge(bg, vr, pts=[(1100, 455), (1100, 270)])
pool = d.box(1130, 220, 180, 100, "2 × 2 pooling<br>→ 8 × 8 tokens<br>ẑ(g), o(g), h(g)", "tok")
d.edge(vr, pool)
d.img(1130, 90, 80, 80, "render_gt", "ẑ(g*)")
d.img(1230, 90, 80, 80, "opacity_gt", "o(g*)")
d.text(870, 140, 200, 50, "rays from the crop's camera,<br>object placed at pose g", 11, "fontStyle=2;")
d.save()

# ------------------------------------------------------------------ 3. joint training
d = D("arch_training")
img = d.img(20, 165, 100, 100, "crop", "training crop I, pose g*")
enc = d.box(220, 100, 150, 60, "encoder ψ (trained)", "obs")
ema = d.box(220, 260, 150, 60, "EMA encoder ψ̄", "obs", extra="dashed=1;")
d.edge(img, enc, pts=[(70, 130)]); d.edge(img, ema, pts=[(70, 290)])
d.edge(enc, ema, label="ψ̄ ← τψ̄ + (1−τ)ψ", dashed=True)
d.img(130, 390, 80, 80, "noisy_poses")
noise = d.box(220, 400, 150, 60, "noisy poses g_t<br>(16 per image)", "plain")
fld = d.box(460, 330, 180, 70, "feature field F_o", "wm")
d.edge(noise, fld, label="render at g_t", pts=[(550, 430)])
den = d.box(740, 120, 180, 70, "denoiser", "den")
d.edge(enc, den, label="u")
d.edge(ema, den, label="z̄", pts=[(680, 290), (680, 170)])
d.edge(fld, den, label="ẑ, o, h (stop-grad)", dashed=True, pts=[(830, 365)])
ld = d.box(990, 120, 200, 70, "denoising loss L_den<br>→ ψ, denoiser", "loss")
d.edge(den, ld)
lf = d.box(740, 450, 200, 70, "field loss E(g*; sg[z̄]), weights w<br>→ field", "loss")
d.edge(fld, lf, label="render at g*", pts=[(550, 485)])
d.edge(ema, lf, label="target (stop-grad)", dashed=True, pts=[(300, 560), (840, 560)])
d.img(960, 445, 75, 75, "render_gt", "ẑ(g*)")
d.img(1045, 445, 75, 75, "residual_gt", "residual")
d.text(990, 230, 220, 80, "L = L_den + λ · E(g*; sg[z̄]), λ = 1<br>the render never receives<br>gradients from L_den", 12, "fontStyle=2;")
d.save()
