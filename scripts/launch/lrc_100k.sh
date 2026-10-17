#!/bin/bash
# Latent render-and-compare denoiser (dit) vs control (dit_nowm) with world model v7c, 100k steps each, in parallel.
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 PYTHONPATH=src
run() { m=$1; python -u -m liepose.exp.cmp.run --root_path $L/experiments/symsolt-lrc "den_mode=[$m]" "wm_kind=[ret]" \
  "wm_dir=[$L/probe/v7c_crop_geo_roll]" "cache_dir=[$D/crop_cache]" "batch_size=[32]" "n_slices=[16]" \
  "train_steps=[100000]" "ckpt_steps=[25000]" > $L/lrc_$m.log 2>&1; echo "$m exit $?"; }
run dit &
sleep 90
run dit_nowm &
wait
echo ALL_DONE
