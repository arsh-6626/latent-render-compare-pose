#!/bin/bash
# SPEED+ latent render-and-compare denoiser: sp_lrc.sh <den_mode> <wm_name>; 100k steps, final ckpt tested on
# the full lightbox + sunlamp splits with the ESA metric (sp_dit.test).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM:-0.2} PYTHONPATH=src
python -u -m liepose.exp.cmp.sp_dit --root_path $L/experiments/speedplus-lrc "den_mode=[$1]" "wm_dir=[$L/probe/$2]" \
  "train_steps=[100000]" "num_workers=[2]" > $L/sp_lrc_$1_$2.log 2>&1
echo "exit $?"
