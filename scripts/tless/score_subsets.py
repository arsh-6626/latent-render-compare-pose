"""BOP19 AR of a scored T-LESS run on object subsets (seen / unseen objects of a held-out-object test), from the
run's own evaluation_results.json: per object, bop_toolkit's recall of each error type; per subset, the recall pooled
over the subset's targets (weights = test_targets_bop19.json instance counts), AR = mean of the MSPD, MSSD and VSD
recalls, as BOP's AR over all objects (checked: the all-object subset reproduces the run's AR).

  DATADIR=... python scripts/tless/score_subsets.py --eval <run dir>/evaluation_results.json --seen 1-20 [--out x.json]
"""

import argparse
import json
import os
from collections import Counter

AR_KEYS = ["mspd_5:50", "mssd_0.05:0.5", "vsd_0.05:0.5"]


def objs(spec):
  lo, hi = map(int, spec.split("-"))
  return list(range(lo, hi + 1))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--eval", required=True)
  p.add_argument("--seen", required=True, help="seen objects, e.g. 1-20; the other target objects are unseen")
  p.add_argument("--out", default=None)
  a = p.parse_args()
  res = json.load(open(a.eval))[0]
  tg = json.load(open(os.path.join(os.environ["DATADIR"], "bop_datasets/tless/test_targets_bop19.json")))
  n = Counter()
  for t in tg:
    n[t["obj_id"]] += t["inst_count"]
  seen = set(objs(a.seen))
  groups = {"all": sorted(n), "seen": sorted(o for o in n if o in seen), "unseen": sorted(o for o in n if o not in seen)}
  out = {"eval": a.eval, "seen_objs": a.seen}
  for g, os_ in groups.items():
    rec = {k: sum(n[o] * res[k]["obj_recalls"][str(o)] for o in os_) / sum(n[o] for o in os_) for k in AR_KEYS}
    out[g] = {"AR": sum(rec.values()) / 3, **rec, "n_targets": sum(n[o] for o in os_), "objs": os_}
    print(f"{g:7s} AR {out[g]['AR']:.3f}  " + "  ".join(f"{k.split('_')[0]} {v:.3f}" for k, v in rec.items())
          + f"  ({out[g]['n_targets']} targets)")
  if a.out:
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
  main()
