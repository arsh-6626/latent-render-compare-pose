#!/bin/bash
# After the SPEED+ token upload: verify sizes, mark it complete; then upload SYMSOL-T crop tokens + world model and mark.
D=/media/kavinder/hdd2/liepose_data; P=/scratch/hpc/visitor/dtu001.visitor/liepose; R=iitd-hpc08:$P/data
LOG=$D/hpc/upload_chain.log; log() { echo "$(date '+%F %T') $*" >> $LOG; }
same() {  # same <local dir> <remote dir>: identical file names and sizes
  [ "$(cd $1 && find . -type f ! -name '.*' -printf '%P %s\n' | sort)" = "$(ssh -o BatchMode=yes iitd-hpc08 "cd $2 && find . -type f ! -name '.*' -printf '%P %s\n' | sort")" ]
}
while pgrep -f "rsync -a --partial $D/speedplus_tokens/" > /dev/null; do sleep 60; done
rsync -a --partial $D/speedplus_tokens/ $R/speedplus_tokens/   # idempotent: finishes anything left
if same $D/speedplus_tokens $P/data/speedplus_tokens && same $D/speedplus_crops_meta $P/data/speedplus_crops; then
  ssh iitd-hpc08 "touch $P/data/speedplus_tokens/.upload_complete" && log "speedplus upload complete"
else log "speedplus upload MISMATCH"; fi
ssh iitd-hpc08 "mkdir -p $P/data/crop_cache"
rsync -a --partial $D/logs/probe/v7c_crop_geo_roll $R/probe/
rsync -a --partial $D/crop_cache/test-5k_p2_{tokens,fg}.npy $D/crop_cache/test-5k_p2_meta.npz \
  $D/crop_cache/train-225k_p2_{tokens,fg}.npy $D/crop_cache/train-225k_p2_meta.npz $R/crop_cache/
ok=1
for f in test-5k_p2_tokens.npy test-5k_p2_fg.npy test-5k_p2_meta.npz train-225k_p2_tokens.npy train-225k_p2_fg.npy train-225k_p2_meta.npz; do
  [ "$(stat -c %s $D/crop_cache/$f)" = "$(ssh iitd-hpc08 "stat -c %s $P/data/crop_cache/$f")" ] || ok=0
done
same $D/logs/probe/v7c_crop_geo_roll $P/data/probe/v7c_crop_geo_roll || ok=0
if [ $ok = 1 ]; then ssh iitd-hpc08 "touch $P/data/crop_cache/.upload_complete" && log "symsolt upload complete"
else log "symsolt upload MISMATCH"; fi
