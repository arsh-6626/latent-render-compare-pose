#!/bin/bash
# Build main.pdf and suppl.pdf with tectonic (XeTeX). Each document reads the other's .aux for cross-references, so the
# first pass of each creates the .aux files and the second pass resolves the references.
cd "$(dirname "$0")"
T=${TECTONIC:-$HOME/.local/bin/tectonic}
for pass in 1 2; do
  for d in suppl main; do $T --keep-intermediates --keep-logs $d.tex > build_$d.log 2>&1 || { echo "$d failed (pass $pass)"; tail -20 build_$d.log; exit 1; }; done
done
grep -h "Warning: Reference\|Warning: Citation" main.log suppl.log | sort -u
echo "main: $(pdfinfo main.pdf | awk '/Pages/{print $2}') pages, suppl: $(pdfinfo suppl.pdf | awk '/Pages/{print $2}') pages"
