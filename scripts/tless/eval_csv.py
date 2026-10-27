"""BOP scoring of an existing T-LESS inference CSV, with the call data/bop/evaluator.Evaluator.evaluate makes
(full_eval: mspd, mssd, vsd, ad, reS, teS on test_targets_bop19.json, n_top -1). For runs whose sampling finished
but whose scoring failed (e.g. no test depth images for VSD on the node); the predictions are not touched.

  DATADIR=... BOP_RENDERER=vispy PYTHONPATH=src python scripts/tless/eval_csv.py \
      --csv <exp>/test_400000/inferences_tless-test.csv --out <local dir>
"""

import argparse
import json
import os
import shutil

from liepose.data.bop.bop_scripts import bop_evaluate, create_tabulate


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--csv", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--meta", default="{}", help="json merged into run.json (exp, step, ...)")
  a = p.parse_args()
  os.makedirs(a.out, exist_ok=True)
  csv = os.path.join(a.out, os.path.basename(a.csv))
  if os.path.abspath(csv) != os.path.abspath(a.csv):
    shutil.copy(a.csv, csv)
  res = bop_evaluate(
    renderer_type=os.environ.get("BOP_RENDERER", "cpp"),
    result_filenames=os.path.abspath(csv),
    results_path="",
    dataset="tless",
    eval_path=os.path.abspath(os.path.join(a.out, "evaluations/")),
    targets_filename="test_targets_bop19.json",
    error_types="mspd,mssd,vsd,ad,reS,teS",
    n_top=-1,
    score_only=False,
    quiet=1,
  )
  json.dump(res, open(os.path.join(a.out, "evaluation_results.json"), "w"), indent=2, ensure_ascii=False)
  table = create_tabulate(res)
  open(os.path.join(a.out, "evaluation_summary.txt"), "w").write(table)
  print(table)
  n = sum(1 for _ in open(csv)) - 1
  json.dump({**json.loads(a.meta), "n_instances": n, "renderer": os.environ.get("BOP_RENDERER", "cpp"),
             "scored_by": "scripts/tless/eval_csv.py"}, open(os.path.join(a.out, "run.json"), "w"), indent=2)


if __name__ == "__main__":
  main()
