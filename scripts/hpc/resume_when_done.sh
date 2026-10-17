#!/bin/bash
# One-shot: wait until no job named $NAME exists, then resubmit it once if its run has no final checkpoint.
#   resume_when_done.sh <job-name> <pbs-script> <ckpt-dir> <final-step>
NAME=$1; SCRIPT=$2; CKPT=$3; FINAL=$4
P=/scratch/hpc/visitor/dtu001.visitor/liepose
LOG=/media/kavinder/hdd2/liepose_data/hpc/resume_when_done.log
while true; do
  out=$(timeout 120 ssh -o BatchMode=yes iitd-hpc08 "echo EXISTS \$(qselect -u dtu001.visitor -N $NAME 2>/dev/null | wc -l); ls -d $CKPT/checkpoint_$FINAL $CKPT/checkpoint_$((FINAL+1)) 2>/dev/null | wc -l | sed 's/^/FINAL /'" 2>/dev/null)
  exists=$(echo "$out" | awk '/^EXISTS/{print $2}'); final=$(echo "$out" | awk '/^FINAL/{print $2}')
  echo "$(date '+%F %T') $NAME exists=$exists final=$final" >> $LOG
  if [ "$exists" = 0 ] && [ -n "$final" ]; then
    if [ "$final" -gt 0 ]; then echo "$(date '+%F %T') $NAME finished training" >> $LOG; exit 0; fi
    jid=$(timeout 60 ssh -o BatchMode=yes iitd-hpc08 "cd $P/code && qsub -N $NAME $SCRIPT" 2>&1 | tail -1)
    echo "$(date '+%F %T') resubmitted $NAME -> $jid" >> $LOG
    exit 0
  fi
  sleep 900
done
