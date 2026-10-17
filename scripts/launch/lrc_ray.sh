#!/bin/bash
# Translation fix: dit + crop-ray hypothesis feature (dit_ray), 100k steps, same recipe/seed as lrc dit.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 PYTHONPATH=src
python -u -m liepose.exp.cmp.run --root_path $L/experiments/symsolt-lrc "den_mode=[dit_ray]" "wm_kind=[ret]" \
  "wm_dir=[$L/probe/v7c_crop_geo_roll]" "cache_dir=[$D/crop_cache]" "batch_size=[32]" "n_slices=[16]" \
  "train_steps=[100000]" "ckpt_steps=[25000]" > $L/lrc_dit_ray.log 2>&1
echo "exit $?"
