#!/bin/bash
# T-LESS stage 1: (a) tless_v1 template WM on the new candidate sets, (b) retrieval WM tless_ret_v1, 30k steps.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 PYTHONPATH=src
python -u -m liepose.exp.jepa_probe.probe_ret_tless --cache $D/tless_tokens --out $L/probe/tless_v1_ceil \
  --template $L/probe/tless_v1 > $L/tless_v1_ceil.log 2>&1
echo "template exit $?"
python -u -m liepose.exp.jepa_probe.probe_ret_tless --cache $D/tless_tokens --out $L/probe/tless_ret_v1 \
  --steps 30000 > $L/tless_ret_v1.log 2>&1
echo "ret exit $?"
