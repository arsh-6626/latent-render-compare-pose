#!/bin/bash
# One-shot: wait until the shared account has <2 scai_q jobs, then qsub ONE job and exit.
# Refuses to submit if a job with the same name already exists (matched exactly with qselect -N).
#   submit_when_free.sh <job-name> <pbs-script-relative-to-code> [extra qsub args]
NAME=$1; SCRIPT=$2; shift 2
P=/scratch/hpc/visitor/dtu001.visitor/liepose
LOG=/media/kavinder/hdd2/liepose_data/hpc/submit_when_free.log
while true; do
  out=$(timeout 120 ssh -o BatchMode=yes iitd-hpc08 "
    echo USED \$(qstat -u dtu001.visitor 2>/dev/null | awk '\$3==\"scai_q\"' | wc -l)
    echo EXISTS \$(qselect -u dtu001.visitor -N $NAME 2>/dev/null | wc -l)" 2>/dev/null)
  used=$(echo "$out" | awk '/^USED/{print $2}')
  exists=$(echo "$out" | awk '/^EXISTS/{print $2}')
  echo "$(date '+%F %T') $NAME used=$used exists=$exists" >> $LOG
  if [ -n "$exists" ] && [ "$exists" -gt 0 ]; then
    echo "$(date '+%F %T') $NAME already exists; not submitting" >> $LOG; exit 0
  fi
  if [ -n "$used" ] && [ "$used" -lt 2 ]; then
    jid=$(timeout 60 ssh -o BatchMode=yes iitd-hpc08 "cd $P/code && qsub -N $NAME $* $SCRIPT" 2>/dev/null | grep pbshpc)
    echo "$(date '+%F %T') submitted $NAME -> $jid" >> $LOG; exit 0
  fi
  sleep 600
done
