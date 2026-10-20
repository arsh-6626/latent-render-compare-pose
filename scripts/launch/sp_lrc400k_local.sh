#!/bin/bash
# SPEED+ 400k-step run on the local GPU (same settings as scripts/hpc/speedplus_lrc400k_dtu004.pbs):
#   scripts/launch/sp_lrc400k_local.sh <den_mode> <wm>      e.g. dit_nowm sp_ret_v1 (the control)
# Re-running resumes from the latest checkpoint.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM:-0.4} PYTHONPATH=src
python -u -m liepose.exp.cmp.sp_dit --root_path $L/experiments/speedplus-lrc "den_mode=[$1]" "wm_dir=[$L/probe/$2]" \
  "train_steps=[400000]" "batch_size=[64]" "n_slices=[64]" "num_workers=[4]" >> $L/sp_lrc400k_$1_$2.log 2>&1
echo "exit $?"
