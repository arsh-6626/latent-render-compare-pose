#!/bin/bash
# Token cache of a BOP dataset for the field world model (the T-LESS recipe; BOP_DS picks the dataset in bop.ds_config):
# LiePose's own crops (export_crops.py) -> frozen DINOv2 tokens (encode_crops.py, torch env) -> crop camera and
# true-centre translation (crop_meta_cache.py). Images are deleted once encoded (--keep_images to keep them).
#   BOP_DS=ycbv N=180000 scripts/tless/build_cache.sh   (TOK / CROPS / META: dirs other than $DATADIR/${BOP_DS}_{tokens,crops,crops_meta})
#   EXPORT_ARGS=--no_mask: crops without the visible mask (test: no mask at all)
set -e
cd "$(dirname "$0")/../.."
D=${DATADIR:-/media/kavinder/hdd2/liepose_data}
NAME=${BOP_DS:?set BOP_DS}
N=${N:-180000}
JAX=${JAX_PY:-$HOME/miniconda3/envs/liepose/bin/python}
TORCH=${TORCH_PY:-$HOME/miniconda3/envs/jepa/bin/python}
export DATADIR=$D CACHE_ROOT_PATH=${CACHE_ROOT_PATH:-$D/bop_cache} PYTHONPATH=src:lib/bop_toolkit BOP_DS=$NAME
C=${CROPS:-$D/${NAME}_crops}; M=${META:-$D/${NAME}_crops_meta}; T=${TOK:-$D/${NAME}_tokens}
mkdir -p $C $M $T
for s in test train; do
  extra=$([ $s = train ] && echo "--n $N" || true)
  geo=$([ $s = train ] && echo "--geometry" || true)  # test: the crop camera only, no 3D ground truth (as tless_crops_meta)
  [ -f $C/$s/.done ] || { $JAX scripts/tless/export_crops.py --split $s $extra ${EXPORT_ARGS} --out $C/$s; touch $C/$s/.done; }
  [ -f $M/$s/.done ] || { $JAX scripts/tless/export_crops.py --split $s $extra --meta_only $geo --out $M/$s; touch $M/$s/.done; }
  [ -f $T/.${s}_encoded ] || { $TORCH scripts/tless/encode_crops.py --crops $C/$s --split $s --out $T; touch $T/.${s}_encoded; }
done
$JAX scripts/tless/crop_meta_cache.py --crops $M --cache $T
[ "$1" = --keep_images ] || rm -rf $C
echo "$(date '+%F %T') $NAME token cache built in $T"
