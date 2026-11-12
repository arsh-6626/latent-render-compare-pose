#!/bin/bash
# sheet.sh <video> <n frames> <out.png> : evenly spaced frames tiled 3 wide (with timestamps)
V=$1; N=${2:-9}; O=$3
[ -f "$V" ] || { echo "missing $V"; exit 1; }
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$V")
TMP=$(mktemp -d)
for i in $(seq 0 $((N-1))); do
  t=$(python3 -c "print(round(($i+0.5)*$D/$N,2))")
  ffmpeg -v error -y -ss $t -i "$V" -frames:v 1 -vf "scale=640:-1,drawtext=text='t=$t':x=8:y=8:fontcolor=yellow:fontsize=22:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" $TMP/f_$(printf %02d $i).png
done
ffmpeg -v error -y -pattern_type glob -i "$TMP/f_*.png" -vf "tile=3x$(( (N+2)/3 )):padding=4" "$O"
rm -rf $TMP
