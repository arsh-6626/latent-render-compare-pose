#!/bin/bash
# LiePose T-LESS baseline: final (400k) checkpoint, LiePose's own test loop + BOP evaluator.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM:-0.15} PYTHONPATH=src BOP_RENDERER=vispy
python -u scripts/tless/test_liepose.py --exp $L/experiments/tless-se3-hpc --step 400000 \
  --out $L/tless_test/liepose_se3_400k > $L/tless_liepose_test.log 2>&1
echo "exit $?"
