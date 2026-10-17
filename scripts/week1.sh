#!/bin/bash
# Week-1 jobs: LiePose SYMSOL-T baseline + latent-energy probe (Gate 1).
# Data and logs live on hdd2 because the root disk is nearly full.
#   bash scripts/week1.sh baseline | probe
set -euo pipefail
cd "$(dirname "$0")/.."

export DATADIR=/media/kavinder/hdd2/liepose_data
export PYTHONPATH=src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0            # RTX 6000 Ada; GPU 1 is the 4 GB A400
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 # three jobs share the Ada
PY="/home/kavinder/miniconda3/envs/liepose/bin/python -u"
LOGS=$DATADIR/logs

case "${1:-}" in
  baseline)
    # Upstream LiePose, SE(3), default config (800k steps, batch 16).
    $PY -m liepose.exp.symsolt.run --root_path $LOGS/experiments/symsolt-score-flat "lie_type=[se3]"
    ;;
  probe)
    # Two probes: with and without the in-batch contrastive term.
    for w in 1.0 0.0; do
      $PY -m liepose.exp.jepa_probe.probe --data $DATADIR/symsolt --out $LOGS/probe/nce$w --w_nce $w
    done
    ;;
  *) echo "usage: $0 baseline|probe"; exit 1 ;;
esac
