#!/bin/bash
# Held-out-object test (diagnostic): retrieval WM trained/selected on T-LESS objects 1-20 only (tless_ret_v1's
# settings otherwise); objects 21-30 are reference views only. Then tless_ret_v1 (all objects) re-scored on the same
# seen / unseen test observations and candidate sets, as the "trained on them" reference.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM:-0.25} PYTHONPATH=src
OBJ=$(seq 1 20)
python -u -m liepose.exp.jepa_probe.probe_ret_tless --cache $D/tless_tokens --out $L/probe/tless_ret_ho20 \
  --steps 30000 --train_objs $OBJ > $L/tless_ret_ho20.log 2>&1
echo "ho20 exit $?"
python -u -m liepose.exp.jepa_probe.probe_ret_tless --cache $D/tless_tokens --out $L/probe/tless_ret_v1 \
  --eval_only --train_objs $OBJ > $L/tless_ret_v1_heldout_eval.log 2>&1
echo "v1 heldout eval exit $?"
