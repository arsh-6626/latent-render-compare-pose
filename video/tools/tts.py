"""Local, offline TTS of script.md with Kokoro (CPU). One wav per beat + durations.json.
  CUDA_VISIBLE_DEVICES= HF_HOME=/media/kavinder/hdd3/larc_video/hf  <manimgl env>/bin/python video/tools/tts.py [beat ids...]
Nothing is sent anywhere: the model weights are downloaded once from the Hugging Face hub; text stays local."""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUD = "/media/kavinder/hdd3/larc_video/audio"
VOICE, SPEED, SR = os.environ.get("LARC_VOICE", "af_heart"), float(os.environ.get("LARC_SPEED", "1.1")), 24000

SUBS = [  # display spelling -> spoken spelling
  (r"\bLaRC\b", "Lark"), (r"\bDINOv2\b", "dino v two"), (r"\bT-LESS\b", "T-less"), (r"\bSPEED\+", "speed plus"),
  (r"\bSPNv3\b", "S P N v three"), (r"\bGDRNPP\b", "G D R N P P"), (r"\bResNet\b", "res net"), (r"\bBOP\b", "bop"),
  (r"\bLiePose\b", "Lee pose"), (r"\bDeepIM\b", "deep I M"), (r"\bCosyPose\b", "cozy pose"), (r"\bMegaPose\b", "mega pose"),
  (r"\bFoundationPose\b", "foundation pose"), (r"\bCAD\b", "cad"), (r"\br o\b", "r oh"), (r"\bw p\b", "w p"),
]


def beats():
  s = open(os.path.join(ROOT, "script.md")).read()
  return re.findall(r"^\[(S\d\d\.\d+)\] (.+)$", s, re.M)


def spoken(t):
  for a, b in SUBS:
    t = re.sub(a, b, t)
  return t


def main():
  from kokoro import KPipeline
  os.makedirs(AUD, exist_ok=True)
  pipe = KPipeline(lang_code="a", device="cpu")
  want = set(sys.argv[1:])
  dpath = os.path.join(AUD, "durations.json")
  dur = json.load(open(dpath)) if os.path.exists(dpath) else {}
  for k, t in beats():
    if want and k not in want:
      continue
    a = np.concatenate([x.numpy() for _, _, x in pipe(spoken(t), voice=VOICE, speed=SPEED)])
    # trim leading / trailing near-silence, keep 60 ms
    idx = np.where(np.abs(a) > 0.01)[0]
    a = a[max(idx[0] - int(0.06 * SR), 0): idx[-1] + int(0.06 * SR)]
    sf.write(os.path.join(AUD, f"{k}.wav"), a, SR)
    dur[k] = len(a) / SR
    print(k, f"{dur[k]:.2f}s", len(t.split()), "words", flush=True)
    json.dump(dur, open(dpath, "w"), indent=1)


if __name__ == "__main__":
  main()
