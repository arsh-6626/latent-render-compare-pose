#!/bin/bash
# SPEED+ stage 2 world model: sp_ret_v1's recipe + geometry tokens (sp_geometry, box-only crop meta).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export DATADIR=$D CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONPATH=src
SP="--cache $D/speedplus_tokens --models_info none --t_scale 1 1 1 --val_split val --test_splits lightbox sunlamp"
python -u -m liepose.exp.jepa_probe.probe_ret_tless $SP --geo --crops $D/speedplus_crops --out $L/probe/sp_geo_v1 --steps 30000 > $L/sp_geo_v1.log 2>&1
echo "geo exit $?"
