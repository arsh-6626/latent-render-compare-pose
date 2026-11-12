"""Headless manimgl launcher: no X display, OpenGL through EGL on the Intel iGPU (Mesa), never the NVIDIA GPUs.
usage: run_manim.py <file.py> <Scene> [manimgl args]"""
import os
import sys

os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import pyglet

pyglet.options["shadow_window"] = False
import moderngl

_orig = moderngl.create_standalone_context


def _egl(*a, **k):
  k.setdefault("backend", "egl")
  k.setdefault("device_index", int(os.environ.get("LARC_EGL_DEVICE", "2")))
  return _orig(*a, **k)


moderngl.create_standalone_context = _egl
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

import manimlib.utils.tex_file_writing as tfw

PREAMBLE = r"""\documentclass[preview]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{xcolor}
"""
TEX_CACHE = Path(os.environ.get("LARC_TEX_CACHE", "/media/kavinder/hdd3/larc_video/texcache"))


def full_tex_to_svg(full_tex, compiler="latex", message=""):
  """TeX -> SVG with tectonic (the TeX engine installed here) and pdftocairo (dvisvgm is not installed)."""
  body = full_tex.split("\\begin{document}", 1)[1].rsplit("\\end{document}", 1)[0]
  doc = PREAMBLE + "\\begin{document}" + body + "\\end{document}\n"
  TEX_CACHE.mkdir(parents=True, exist_ok=True)
  key = TEX_CACHE / (hashlib.sha256(doc.encode()).hexdigest()[:24] + ".svg")
  if key.exists():
    return key.read_text()
  with tempfile.TemporaryDirectory() as d:
    tp = Path(d, "w.tex")
    tp.write_text(doc)
    r = subprocess.run(["tectonic", "-X", "compile", "--untrusted", "--outdir", d, str(tp)], capture_output=True, text=True)
    if r.returncode != 0:
      raise tfw.LatexError(r.stderr[-1500:])
    subprocess.run(["pdftocairo", "-svg", str(Path(d, "w.pdf")), str(Path(d, "w.svg"))], check=True)
    svg = Path(d, "w.svg").read_text()
  key.write_text(svg)
  return svg


tfw.full_tex_to_svg = full_tex_to_svg
from manimlib.__main__ import main

sys.argv = ["manimgl"] + sys.argv[1:]
main()
