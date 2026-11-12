#!/bin/bash
# render.sh <file.py> <Scene> [low|hd]   -> /media/kavinder/hdd3/larc_video/render/<Scene>.mp4
E=/media/kavinder/hdd3/larc_video/envs/manimgl
export PATH=$E/bin:$PATH CUDA_VISIBLE_DEVICES=
cd /home/kavinder/ARNABI_ARSH/LARp/video/scenes
OUTD=/media/kavinder/hdd3/larc_video/render/$3; mkdir -p $OUTD
if [ "$3" = "hd" ]; then Q="--hd"; else Q="-l"; fi
$E/bin/python run_manim.py $1 $2 -w $Q --video_dir $OUTD --file_name $2 2>&1 | grep -vE "it/s\]|pkg_resources|^\s*$" | tail -15
