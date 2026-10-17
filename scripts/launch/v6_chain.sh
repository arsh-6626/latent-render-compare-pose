#!/bin/bash
cd /home/kavinder/ARNABI_ARSH/liepose-diffusion
PY=~/miniconda3/envs/liepose/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
COMMON="--cache /media/kavinder/hdd2/liepose_data/dino_cache --pool 2 --train_split train-225k --init /media/kavinder/hdd2/liepose_data/logs/probe/v4_trans --steps 15000 --batch_size 32  --n_uni 11 --n_near 12 --near_min_deg 0 --near_max_deg 30 --soft_sigma 15 --n_tuni 4 --n_tnear 4 --tnear_std 0.1  --sigma_t 0.1 --lr 1e-4 --eval_every 5000 --eval_per_shape 20 --eval_rand 1024"
$PY -m liepose.exp.jepa_probe.probe_frozen $COMMON --pairs /media/kavinder/hdd2/liepose_data/pairs --w_pair 1.0 --pair_batch 8 --out /media/kavinder/hdd2/liepose_data/logs/probe/v6_pair > /media/kavinder/hdd2/liepose_data/logs/probe_v6_pair.log 2>&1
$PY -m liepose.exp.jepa_probe.probe_frozen $COMMON --out /media/kavinder/hdd2/liepose_data/logs/probe/v6_nopair > /media/kavinder/hdd2/liepose_data/logs/probe_v6_nopair.log 2>&1
for w in v6_pair v6_nopair; do
  $PY scripts/oracle/ceiling_wm.py --wm /media/kavinder/hdd2/liepose_data/logs/probe/$w --cache /media/kavinder/hdd2/liepose_data/dino_cache --ceiling /media/kavinder/hdd2/liepose_data/oracle/ceiling.npz --out /media/kavinder/hdd2/liepose_data/oracle/ceiling_wm_$w.json > /media/kavinder/hdd2/liepose_data/logs/ceiling_wm_$w.log 2>&1
done
for w in v4_trans v6_pair v6_nopair; do
  $PY scripts/oracle/pair_fit.py --wm /media/kavinder/hdd2/liepose_data/logs/probe/$w --pairs /media/kavinder/hdd2/liepose_data/pairs --ceiling /media/kavinder/hdd2/liepose_data/oracle/ceiling.npz --cache /media/kavinder/hdd2/liepose_data/dino_cache --out /media/kavinder/hdd2/liepose_data/oracle/pair_fit_$w.json > /media/kavinder/hdd2/liepose_data/logs/pair_fit_$w.log 2>&1
done
echo CHAIN_DONE
