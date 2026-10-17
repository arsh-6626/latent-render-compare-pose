#!/bin/bash
# Local watcher: submit the week-2 cmp jobs to scai_q as the shared account's 2-job limit frees up.
# A COND counts as done once its eval summary.json exists; a COND whose job ended without one
# (e.g. hit the 24 h cap) is resubmitted and resumes from its last checkpoint.
# Jobs are matched by exact name with qselect -N (qstat truncates names, which once caused duplicates).
P=/scratch/hpc/visitor/dtu001.visitor/liepose
ORDER="compare none pooled"
LOG=/media/kavinder/hdd2/liepose_data/hpc/watch_submit.log

while true; do
  out=$(timeout 120 ssh -o BatchMode=yes iitd-hpc08 "
    n=\$(qstat -u dtu001.visitor 2>/dev/null | awk '\$3==\"scai_q\"' | wc -l)
    echo SLOTS_USED \$n
    for c in $ORDER; do
      if ls $P/logs/experiments/symsolt-cmp/*cond_mode=\$c*/eval_*/summary.json >/dev/null 2>&1; then echo DONE \$c;
      elif [ -n \"\$(qselect -u dtu001.visitor -N cmp-\$c 2>/dev/null)\" ]; then echo ACTIVE \$c;
      else echo PENDING \$c; fi
    done" 2>/dev/null | grep -E "^(SLOTS_USED|DONE|ACTIVE|PENDING)")
  echo "$(date '+%F %T') $(echo $out | tr '\n' ' ')" >> $LOG
  used=$(echo "$out" | awk '/SLOTS_USED/{print $2}')
  pending=$(echo "$out" | awk '/PENDING/{print $2}')
  if [ -z "$pending" ] && ! echo "$out" | grep -q ACTIVE && [ -n "$used" ]; then
    echo "$(date '+%F %T') all cmp jobs done" >> $LOG
    exit 0
  fi
  if [ -n "$used" ] && [ "$used" -lt 2 ]; then
    for c in $pending; do
      [ "$used" -ge 2 ] && break
      jid=$(timeout 60 ssh -o BatchMode=yes iitd-hpc08 "cd $P/code && qsub -N cmp-$c -v COND=$c scripts/hpc/cmp_symsolt_scai.pbs" 2>/dev/null | grep pbshpc)
      echo "$(date '+%F %T') submitted $c -> $jid" >> $LOG
      used=$((used + 1))
    done
  fi
  sleep 600
done
