#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/LARp/video/scenes
L=/media/kavinder/hdd3/larc_video/render/hd_log; mkdir -p $L
run() { ./render.sh $1 $2 hd > $L/$2.log 2>&1; echo "$2 done $(date +%T)" >> $L/progress.txt; }
(run s07_field.py S07; run s01_problem.py S01; run s03_diffusion.py S03; run s05_render_compare.py S05; run s09_loop.py S09; run s11_flow.py S11; run s13_discrimination.py S13) &
(run s02_group.py S02; run s04_openloop.py S04; run s06_tokens.py S06; run s08_fit.py S08; run s10_select.py S10; run s12_results.py S12; run s14_synthesis.py S14) &
wait
echo ALL >> $L/progress.txt
