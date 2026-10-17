"""Oracle check with frozen DINOv2 ViT-S/14 patch tokens (run in a torch-CUDA env, e.g. `jepa`).

  python scripts/oracle/feats_dino.py --bank $DATADIR/oracle/bank.npz --out $DATADIR/oracle/results_dino.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from metrics import print_table, score  # noqa: E402


@torch.no_grad()
def main():
  p = argparse.ArgumentParser()
  p.add_argument("--bank", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--model", default="dinov2_vits14")
  a = p.parse_args()

  bank = dict(np.load(a.bank))
  images = bank["images"]
  n_bank = len(bank["R_bank"])
  net = torch.hub.load("facebookresearch/dinov2", a.model).cuda().eval()
  mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

  results = {}
  for layer_name in ["patch", "cls"]:
    dists = []
    for s in range(images.shape[0]):
      feats = []
      for i in range(0, images.shape[1], 128):
        x = torch.from_numpy(images[s, i : i + 128]).cuda().permute(0, 3, 1, 2).float() / 255
        out = net.forward_features((x - mean) / std)
        f = out["x_norm_patchtokens"] if layer_name == "patch" else out["x_norm_clstoken"][:, None]
        feats.append(f.flatten(1))
      f = torch.cat(feats)
      b, q = f[:n_bank], f[n_bank:]
      d = torch.cdist(q, b) ** 2 / f.shape[1]
      dists.append(d.cpu().numpy())
    name = f"{a.model}_{layer_name}"
    results[name] = score(np.stack(dists), bank)
    print_table(name, results[name])

  with open(a.out, "w") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  main()
