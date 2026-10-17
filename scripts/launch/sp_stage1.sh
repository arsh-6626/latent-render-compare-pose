#!/bin/bash
# SPEED+ stage 1, after the T-LESS WM run frees its GPU share: (a) speedplus_v1 template WM on the new candidates,
# (b) retrieval WM sp_ret_v1 (no symmetries, t_scale 1, ckpt on synthetic val, tested on lightbox + sunlamp).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
while pgrep -f "probe_ret_tless --cache $D/tless_tokens" > /dev/null; do sleep 60; done
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 PYTHONPATH=src
SP="--cache $D/speedplus_tokens --models_info none --t_scale 1 1 1 --val_split val --test_splits lightbox sunlamp"
python -u -m liepose.exp.jepa_probe.probe_ret_tless $SP --out $L/probe/speedplus_v1_ceil --template $L/probe/speedplus_v1 > $L/speedplus_v1_ceil.log 2>&1
echo "template exit $?"
python -u -m liepose.exp.jepa_probe.probe_ret_tless $SP --out $L/probe/sp_ret_v1 --steps 30000 > $L/sp_ret_v1.log 2>&1
echo "ret exit $?"
