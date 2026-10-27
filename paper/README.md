# Paper: Latent Render-and-Compare (CVPR 2027 submission draft)

Template: CVPR author kit (https://github.com/cvpr-org/author-kit, CVPR 2026 release; `\confYear` set to 2027).

- `main.tex` — submission (8 pages + references), sections in `sec/0_*.tex` … `sec/6_*.tex`, teaser in `fig/teaser.tex`.
- `suppl.tex` — supplementary material (`sec/X_suppl.tex`), compiled separately; cross-references to/from `main` via `xr-hyper`.
- `main.bib` — bibliography; entries marked `% VERIFY` have not yet been checked against the source paper.
- Build: `./build.sh` (tectonic/XeTeX, two passes of both documents). With pdflatex/Overleaf the XeTeX font block in
  `preamble.tex` is skipped automatically.

Conventions in this draft:
- Every result cell is empty (`\pend`). Rows with published numbers carry a `%` comment naming the source table.
- No result appears in the prose; sentences that depend on results are `\TODO{...}` (red).
- Rows and figures that use ground truth (oracle selection, oracle tokens, flip timing, world-model discrimination) are diagnostics
  and labelled as such.
- Disclosures (GT crops, object identity for the bank, VSD renderer, 64 vs 256 noisy poses, EMA/live weight choice, SYMSOL-T world
  model development) are in the supplement, Sec. F.
