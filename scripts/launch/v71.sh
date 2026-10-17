#!/bin/bash
# After the v7 ablation chain: v7.1 = v7 + residual + diff + rank losses, 40k from v7_ret_geo_roll (lr 1e-4, fresh cosine).
until grep -q ALL_DONE /tmp/claude-1000/-home-kavinder-ARNABI-ARSH/41892fcf-246d-49cc-9e4f-f265e33a06d8/tasks/bqsp9kodz.output; do sleep 60; done
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate liepose
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
D=/media/kavinder/hdd2/liepose_data; L=$D/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONPATH=src
python -m liepose.exp.jepa_probe.probe_ret --cache $D/dino_cache --ceiling $D/oracle/ceiling.npz --pairs $D/pairs \
  --rolls $D/rolls_wide --geo --init $L/probe/v7_ret_geo_roll --steps 40000 --lr 1e-4 --resid --w_diff 1 --w_rank 1 \
  --out $L/probe/v71_resid_diff_rank > $L/probe_v71_resid_diff_rank.log 2>&1
echo "v71 exit $?"
