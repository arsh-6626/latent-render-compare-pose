"""Architecture diagrams as draw.io files (plain mxGraph XML; open in draw.io / diagrams.net, File > Open from device).
Run: python arch_drawio.py -> LARp/paper/fig/arch/*.drawio
Content follows the method section (Sec. 3) and the code (tless_field_dit.py, sp_field_dit.py, tless_field.py)."""
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

  def box(self, x, y, w, h, text, kind="plain", rounded=True, extra=""):
    i = self._id()
    st = f"rounded={int(rounded)};whiteSpace=wrap;html=1;fillColor={COL[kind]};{FONT}arcSize=12;{extra}"
    self.cells.append(f'<mxCell id="{i}" value="{html.escape(text)}" style="{st}" vertex="1" parent="1">'
                      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

  def group(self, x, y, w, h, title, kind="white"):
    return self.box(x, y, w, h, title, kind, extra="dashed=1;verticalAlign=top;fontStyle=2;opacity=60;")

  def text(self, x, y, w, h, text, size=12, extra=""):
    i = self._id()
    st = f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;fontFamily=Times New Roman;fontSize={size};{extra}"
    self.cells.append(f'<mxCell id="{i}" value="{html.escape(text)}" style="{st}" vertex="1" parent="1">'
                      f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

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
           f'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1400" '
           f'pageHeight="700" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
           f'</mxGraphModel></diagram></mxfile>')
    open(f"{OUT}/{self.name}.drawio", "w").write(xml)
    print("saved", f"{OUT}/{self.name}.drawio")


# ------------------------------------------------------------------ 1. overview: latent render-and-compare inside the sampler
d = D("arch_overview")
crop = d.box(20, 250, 120, 70, "detected crop I<br>(box only)", "obs")
enc = d.box(190, 200, 140, 60, "encoder ψ<br>(DINOv2-S, trained)", "obs")
ema = d.box(190, 300, 140, 60, "EMA encoder ψ̄", "obs", extra="dashed=1;")
d.edge(crop, enc); d.edge(crop, ema)
zobs = d.box(380, 310, 120, 40, "z̄ (64 tokens)", "tok")
uext = d.box(380, 210, 120, 40, "u (64 tokens)", "tok")
d.edge(enc, uext); d.edge(ema, zobs)
d.group(540, 40, 300, 200, "world model (per-object feature field)", "wm")
fld = d.box(565, 80, 250, 70, "field F_o: tri-planes + MLP<br>volume render at g_t", "wm")
app = d.box(565, 170, 110, 50, "appearance a(z̄)", "wm")
outs = d.box(690, 170, 125, 50, "ẑ(g_t), o(g_t), h(g_t)", "tok")
d.edge(zobs, app, pts=[(620, 330)])
d.edge(app, fld, pts=[(620, 160)])
d.edge(fld, outs, pts=[(752, 160)])
tk = d.box(890, 260, 190, 80, "tokens [z̄, ẑ, ẑ−z̄, w, h] + u<br>(per patch, 64)", "tok")
d.edge(outs, tk, label="stop-grad", dashed=True, pts=[(985, 195)])
d.edge(zobs, tk, pts=[(700, 330)])
d.edge(uext, tk, pts=[(860, 230), (860, 280)])
den = d.box(890, 400, 190, 70, "denoiser (4 adaLN DiT blocks)<br>cond: t, r_t", "den")
d.edge(tk, den)
step = d.box(560, 400, 220, 70, "reverse step on SE(3)<br>g_t → g_{t−1}", "den")
d.edge(den, step, label="ε̂")
gt = d.box(380, 410, 120, 50, "hypothesis g_t", "plain")
d.edge(step, gt)
d.edge(gt, fld, label="render at g_t", pts=[(440, 115)])
d.text(380, 470, 400, 30, "repeated for 100 steps, 16 chains in parallel", 12, "fontStyle=2;")
sel = d.box(1130, 400, 150, 70, "select one of 16<br>(consensus / energy E)", "loss")
d.edge(den, sel, pts=[(1100, 435)])
res = d.box(1130, 520, 150, 50, "pose ĝ", "plain")
d.edge(sel, res)
d.save()

# ------------------------------------------------------------------ 2. the feature field
d = D("arch_field")
x = d.box(20, 140, 130, 60, "point x on a ray<br>(object frame)", "plain")
tri = d.box(200, 40, 170, 80, "tri-planes of object o<br>3 × 128² × 32<br>(bilinear, summed)", "wm")
four = d.box(200, 220, 170, 60, "Fourier features of x<br>(4 octaves)", "wm")
d.edge(x, tri, pts=[(85, 80)]); d.edge(x, four, pts=[(85, 250)])
mlp = d.box(420, 130, 150, 80, "MLP 2 × 128, GELU<br>→ latent h(x)", "wm")
d.edge(tri, mlp, pts=[(495, 80)]); d.edge(four, mlp, pts=[(495, 250)])
dens = d.box(620, 40, 190, 60, "density σ = softplus(ℓ(h) − 1)<br>(0 outside the box)", "wm")
head = d.box(620, 220, 190, 60, "head 2 × 256 (FiLM by a)<br>→ feature f_obj(x)", "wm")
d.edge(mlp, dens, pts=[(715, 170)]); d.edge(mlp, head, pts=[(715, 170)])
apc = d.box(620, 320, 190, 60, "appearance a = GELU(Dense(μ, σ of z̄))", "obs")
d.edge(apc, head)
vr = d.box(870, 120, 200, 100, "volume rendering<br>16 × 16 rays, 32 samples<br>f = o·f_obj + (1 − o)·f_bg", "wm")
d.edge(dens, vr, pts=[(970, 70)]); d.edge(head, vr, pts=[(970, 250)])
bg = d.box(870, 320, 200, 60, "background f_bg = bg(a)", "obs")
d.edge(apc, bg); d.edge(bg, vr, pts=[(1100, 350), (1100, 170)])
pool = d.box(1120, 120, 180, 100, "2 × 2 pooling<br>→ 8 × 8 tokens<br>ẑ(g), o(g), h(g)", "tok")
d.edge(vr, pool)
d.text(870, 40, 200, 50, "rays from the crop's camera,<br>object placed at pose g", 11, "fontStyle=2;")
d.save()

# ------------------------------------------------------------------ 3. joint training
d = D("arch_training")
img = d.box(20, 180, 140, 70, "training crop I<br>pose g*", "obs")
enc = d.box(220, 100, 150, 60, "encoder ψ (trained)", "obs")
ema = d.box(220, 260, 150, 60, "EMA encoder ψ̄", "obs", extra="dashed=1;")
d.edge(img, enc, pts=[(90, 130)]); d.edge(img, ema, pts=[(90, 290)])
d.edge(enc, ema, label="ψ̄ ← τψ̄ + (1−τ)ψ", dashed=True)
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
d.edge(ema, lf, label="target (stop-grad)", dashed=True, pts=[(300, 540), (840, 540)])
d.text(990, 230, 220, 80, "L = L_den + λ · E(g*; sg[z̄]), λ = 1<br>the render never receives<br>gradients from L_den", 12, "fontStyle=2;")
d.save()
