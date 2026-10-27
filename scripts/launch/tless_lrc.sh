#!/bin/bash
# T-LESS latent render-and-compare denoiser: tless_lrc.sh <den_mode> <wm_name>; 100k steps, final ckpt tested with
# LiePose's TestDataset + BOP evaluator (AR on test_targets_bop19; VSD via the vispy renderer).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM:-0.2} PYTHONPATH=src BOP_RENDERER=vispy
python -u -m liepose.exp.cmp.tless_dit --root_path $L/experiments/tless-lrc "den_mode=[$1]" "wm_dir=[$L/probe/$2]" \
  "train_steps=[100000]" "num_workers=[2]" > $L/tless_lrc_$1_$2.log 2>&1
echo "exit $?"
