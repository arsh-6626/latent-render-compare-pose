#!/bin/bash
# SPEED+ best-of-N (world-model energy selection) test of a trained sp_dit run: scripts/launch/sp_bestofn.sh <wm> [n]
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs; WM=$1; N=${2:-16}
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM:-0.3} PYTHONPATH=src
python -u scripts/speedplus/test_lrc_bestofn.py --exp "$L/experiments/speedplus-lrc/train_steps=100000+seed=42+den_mode=dit+wm=$WM" \
  --step 100000 --n $N --out $L/speedplus_test/lrc_dit_${WM}_100k_best$N > $L/sp_bestofn_${WM}_$N.log 2>&1
echo "exit $?"
