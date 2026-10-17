#!/bin/bash
# K=16 samples per image for the 100k dit (model C) run; errors.npz keeps per-sample poses + WM self-energy.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.42 PYTHONPATH=src
python -u -m liepose.exp.cmp.run --root_path $L/experiments/symsolt-lrc "den_mode=[dit]" "wm_kind=[ret]" \
  "wm_dir=[$L/probe/v7c_crop_geo_roll]" "cache_dir=[$D/crop_cache]" "batch_size=[32]" "n_slices=[16]" \
  "train_steps=[100000]" "ckpt_steps=[25000]" "eval_samples=[16]" "eval_tag=[_k16]" "skip_train=[True]" > $L/lrc_dit_k16.log 2>&1
echo "exit $?"
