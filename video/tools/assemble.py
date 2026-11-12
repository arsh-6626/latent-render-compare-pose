"""Assemble the final MP4: concatenate the rendered scenes, place every narration beat at the time the scene logged for it,
normalise loudness, encode H.264 1080p + AAC.
  <manimgl env>/bin/python video/tools/assemble.py [hd|low]"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

Q = sys.argv[1] if len(sys.argv) > 1 else "hd"
W = "/media/kavinder/hdd3/larc_video"
REN, AUD, TIM = f"{W}/render/{Q}", f"{W}/audio", f"{W}/render/timing"
OUT = "/home/kavinder/ARNABI_ARSH/LARp/video/out/larc_explainer.mp4" if Q == "hd" else f"{W}/larc_explainer_{Q}.mp4"
SR = 24000
SCENES = [f"S{i:02d}" for i in range(1, 15)]


def dur(p):
  return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]).decode())


offs, total, report = {}, 0.0, []
for s in SCENES:
  offs[s] = total
  total += dur(f"{REN}/{s}.mp4")
audio = np.zeros(int(total * SR) + SR, np.float32)
for s in SCENES:
  tl = json.load(open(f"{TIM}/{s}.json"))
  vd = dur(f"{REN}/{s}.mp4")
  for k, t0 in tl["beats"].items():
    a, sr = sf.read(f"{AUD}/{k}.wav", dtype="float32")
    assert sr == SR
    i = int(round((offs[s] + t0) * SR))
    audio[i:i + len(a)] += a
    end = t0 + len(a) / SR
    report.append((k, offs[s] + t0, offs[s] + end, end <= vd + 1e-3))
    if end > vd + 1e-3:
      print(f"WARNING {k} ends at {end:.2f}s after its scene ({vd:.2f}s)")
# overlap check
report.sort(key=lambda r: r[1])
for a, b in zip(report[:-1], report[1:]):
  if b[1] < a[2] - 1e-3:
    print(f"WARNING overlap {a[0]} / {b[0]}")
peak = np.abs(audio).max()
audio = audio / max(peak, 1e-6) * 0.9
sf.write(f"{W}/narration_{Q}.wav", audio[:int(total * SR)], SR)
json.dump({"scene_offsets": offs, "total": total, "beats": report}, open(f"{W}/assembly_{Q}.json", "w"), indent=1)

lst = f"{W}/concat_{Q}.txt"
open(lst, "w").write("".join(f"file '{REN}/{s}.mp4'\n" for s in SCENES))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", f"{W}/narration_{Q}.wav",
       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
       "-r", "30", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000", "-c:a", "aac", "-b:a", "192k",
       "-movflags", "+faststart", "-shortest", OUT]
subprocess.run(cmd, check=True)
print("wrote", OUT, f"{total:.1f}s")
for s in SCENES:
  print(s, f"{offs[s]:.1f}")
