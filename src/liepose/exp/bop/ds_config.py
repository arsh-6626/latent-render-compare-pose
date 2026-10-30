"""Per-dataset constants of the BOP experiments, selected by the environment variable BOP_DS (default "tless", which
reproduces the T-LESS-only code exactly). Read by bop.run.build_explorations (LiePose's datasets and translation
parametrisation, so also tless_dit / tless_field_dit / export_crops) and by jepa_probe.tless_field (object count,
box radii, translation parametrisation of the field's geometry).

tran_scale / tran_offset: LiePose's translation_scaled = raw / tran_scale - tran_offset, raw = [dx / bbox_w, dy / bbox_h,
z / focal_ratio]; for a new dataset from scripts/tless/tran_stats.py on its training sets (the same procedure checked
against the T-LESS constants), and the same values for LiePose and for every model trained on that dataset.
"""

import os

DATASETS = {
  "tless": dict(
    n_obj=30,
    train_datasets=["tless-train_pbr", "tless-train_primesense"],
    test_datasets=["tless-test_primesense"],
    tran_scale=[0.044119, 0.054692, 0.238700],
    tran_offset=[0.11813018, 0.08893179, 2.5791266],
    models_info="bop_datasets/tless/models_eval/models_info.json",
    tokens="tless_tokens",
  ),
  "ycbv": dict(
    n_obj=21,
    train_datasets=["ycbv-train_pbr"],  # PBR only for now (train_real is not on the cluster yet)
    test_datasets=["ycbv-test"],
    # scripts/tless/tran_stats.py --datasets ycbv-train_pbr --n 5000 --augment 0: robust scale (1.4826 MAD), median / MAD
    tran_scale=[0.043576, 0.062371, 0.370406],
    tran_offset=[0.060127, 0.045836, 2.649923],
    models_info="bop_datasets/ycbv/models_eval/models_info.json",
    tokens="ycbv_tokens",
  ),
}

NAME = os.environ.get("BOP_DS", "tless")
assert NAME in DATASETS, f"BOP_DS={NAME!r} is not one of {sorted(DATASETS)}"
DS = DATASETS[NAME]
