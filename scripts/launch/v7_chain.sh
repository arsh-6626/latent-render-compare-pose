#!/bin/bash
# v7 ablations: retrieval reference (1), + rolls (4), + geometry (2). Two at a time, then the third.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONPATH=src
COMMON="--cache $D/dino_cache --ceiling $D/oracle/ceiling.npz --pairs $D/pairs --init $L/probe/v6_pair --steps 20000"
run() { name=$1; shift; python -m liepose.exp.jepa_probe.probe_ret $COMMON --out $L/probe/$name "$@" > $L/probe_$name.log 2>&1; echo "$name exit $?"; }
run v7_ret_geo_roll --rolls $D/rolls_wide --geo &
run v7_ret &
wait
run v7_ret_roll --rolls $D/rolls_wide
echo ALL_DONE
