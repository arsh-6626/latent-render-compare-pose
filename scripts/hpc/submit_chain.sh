#!/bin/bash
# Submit jobs in order, one at a time: keep retrying qsub for the next job until the cluster accepts it.
# Skips a job if one with the same exact name already exists (qselect -N) or its eval summary exists.
#   submit_chain.sh <script> <name1>:<MODE1> <name2>:<MODE2> ...
SCRIPT=$1; shift
P=/scratch/hpc/visitor/dtu001.visitor/liepose
LOG=/media/kavinder/hdd2/liepose_data/hpc/submit_chain.log
for item in "$@"; do
  NAME=${item%%:*}; MODE=${item##*:}
  while true; do
    out=$(timeout 120 ssh -o BatchMode=yes iitd-hpc08 "
      if [ -n \"\$(qselect -u dtu001.visitor -N $NAME 2>/dev/null)\" ]; then echo EXISTS;
      elif ls $P/logs/experiments/symsolt-wmden/*den_mode=$MODE+*/eval_*/summary.json >/dev/null 2>&1; then echo DONE;
      else cd $P/code && qsub -N $NAME -v MODE=$MODE $SCRIPT 2>&1; fi" 2>&1 | grep -v "esac\|^}" | tail -1)
    echo "$(date '+%F %T') $NAME ($MODE): $out" >> $LOG
    case "$out" in *pbshpc*|EXISTS|DONE) break ;; esac
    sleep 900
  done
done
echo "$(date '+%F %T') chain finished" >> $LOG
