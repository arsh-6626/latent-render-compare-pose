#!/bin/bash
# Local queue keeper for the 400k-step runs. Every 15 min, per account (scai_q: 2 jobs incl. running), walks an ordered
# job list and submits the first unfinished jobs that are not queued, up to 2. A job is finished when its final
# checkpoint exists; until then each resubmission resumes from the latest checkpoint. A job waits for its data marker
# (written after the upload) and is submitted at most MAXSUB times (a crashing job must not loop).
#   nohup scripts/hpc/queue_chain.sh > /dev/null 2>&1 &      log: $LOG
P=/scratch/hpc/visitor/dtu001.visitor/liepose; W=/scratch/hpc/visitor/dtu004.visitor/lrc
LOG=/media/kavinder/hdd2/liepose_data/hpc/queue_chain.log; CNT=/media/kavinder/hdd2/liepose_data/hpc/queue_chain.counts
MAXSUB=5; mkdir -p $(dirname $LOG); touch $CNT
PBS='export PATH=/opt/pbs/2024.1.5/bin:$PATH'
# name|script|vars|final checkpoint|data marker
A1=(
"tl400-dit|tless_lrc400k_scai.pbs|MODE=dit|$P/logs/experiments/tless-lrc/train_steps=400000+seed=42+den_mode=dit+wm=tless_ret_v1/ckpt/checkpoint_400000|$P/data/tless_tokens/train_tokens.npy"
"tl400-nowm|tless_lrc400k_scai.pbs|MODE=dit_nowm|$P/logs/experiments/tless-lrc/train_steps=400000+seed=42+den_mode=dit_nowm+wm=tless_ret_v1/ckpt/checkpoint_400000|$P/data/tless_tokens/train_tokens.npy"
"ss400-dit|symsolt_lrc400k_scai.pbs|MODE=dit|$P/logs/experiments/symsolt-lrc/train_steps=400000+n_slices=64+seed=42+cond_mode=compare+cond_gate_t=-1+den_mode=dit/ckpt/checkpoint_400000|$P/data/crop_cache/.upload_complete"
"ss400-nowm|symsolt_lrc400k_scai.pbs|MODE=dit_nowm|$P/logs/experiments/symsolt-lrc/train_steps=400000+n_slices=64+seed=42+cond_mode=compare+cond_gate_t=-1+den_mode=dit_nowm/ckpt/checkpoint_400000|$P/data/crop_cache/.upload_complete"
)
A4=(
"sp400-geo|speedplus_lrc400k_dtu004.pbs|MODE=dit,WM=sp_geo_v1|$W/experiments/speedplus-lrc/train_steps=400000+seed=42+den_mode=dit+wm=sp_geo_v1/ckpt/checkpoint_400000|$P/data/speedplus_tokens/.upload_complete"
"sp400-nowm|speedplus_lrc400k_dtu004.pbs|MODE=dit_nowm,WM=sp_ret_v1|$W/experiments/speedplus-lrc/train_steps=400000+seed=42+den_mode=dit_nowm+wm=sp_ret_v1/ckpt/checkpoint_400000|$P/data/speedplus_tokens/.upload_complete"
)
h1() { timeout 120 ssh -o BatchMode=yes iitd-hpc08 "$1"; }
h4() { timeout 150 ssh -o BatchMode=yes yashdeep@10.141.90.34 "ssh -o BatchMode=yes iitd-hpc $(printf %q "$1")"; }
log() { echo "$(date '+%F %T') $*" >> $LOG; }

keep() {  # keep <ssh-fn> <user> <jobs...>
  local H=$1 U=$2; shift 2
  local q; q=$($H "$PBS; qstat -u $U 2>/dev/null | awk 'NR>5{print \$4}'") || { log "$U unreachable"; return; }
  local n; n=$(echo "$q" | grep -c .)
  local all_done=1
  for j in "$@"; do
    IFS='|' read -r name script vars final marker <<< "$j"
    if echo "$q" | grep -qx "${name:0:10}"; then all_done=0; continue; fi
    st=$($H "[ -e '$final' ] && echo DONE || { [ -e '$marker' ] && echo READY || echo NODATA; }") || { log "$U check failed"; return; }
    [ "$st" = DONE ] && continue
    all_done=0
    [ "$st" = NODATA ] && { log "$name waiting for data"; continue; }
    [ "$n" -ge 2 ] && continue
    c=$(grep -c "^$name$" $CNT)
    [ "$c" -ge $MAXSUB ] && { log "$name NOT resubmitted: $c submissions already (check its log)"; continue; }
    jid=$($H "$PBS; cd $P/code_lrc/scripts/hpc && qsub -N $name -v $vars $script" 2>&1 | tail -1)
    echo "$name" >> $CNT; n=$((n + 1)); log "submitted $name ($((c + 1))/$MAXSUB) -> $jid"
  done
  [ $all_done = 1 ] && log "$U ALL_DONE"
}

# the two T-LESS jobs submitted by hand count as their first submission
grep -q "^tl400-dit$" $CNT || printf "tl400-dit\ntl400-nowm\n" >> $CNT
while true; do
  keep h1 dtu001.visitor "${A1[@]}"
  keep h4 dtu004.visitor "${A4[@]}"
  if grep -q "dtu001.visitor ALL_DONE" $LOG && grep -q "dtu004.visitor ALL_DONE" $LOG; then log "EVERYTHING_DONE"; exit 0; fi
  sleep 900
done
