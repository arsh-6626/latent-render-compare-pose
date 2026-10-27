#!/bin/bash
# Keep a resumable dtu001 job going across the 24 h walltime: whenever no job named $NAME exists and the run has no
# test result yet (<exp-dir>/test_*/esa.json), resubmit it (training resumes from the latest checkpoint; a finished
# run only tests). Runs locally, e.g. under nohup.
#   resume_loop.sh <job-name> <exp-dir> <max-submits> <qsub args...>
#   resume_loop.sh ft400-wm "$P/logs/experiments/speedplus-lrc/train_steps=400000+seed=42+den_mode=dit_ft+wm=sp_geo_v1" 5 \
#       -v MODE=dit_ft speedplus_ft400k_scai.pbs
NAME=$1; EXP=$2; MAX=$3; shift 3
P=/scratch/hpc/visitor/dtu001.visitor/liepose
LOG=/media/kavinder/hdd2/liepose_data/hpc/resume_loop_$NAME.log
n=0
while true; do
  out=$(timeout 120 ssh -o BatchMode=yes iitd-hpc08 "source /etc/profile; echo EXISTS \$(qselect -u dtu001.visitor -N $NAME 2>/dev/null | wc -l); ls '$EXP'/test_*/esa.json 2>/dev/null | wc -l | sed 's/^/DONE /'" 2>/dev/null)
  exists=$(echo "$out" | awk '/^EXISTS/{print $2}'); done=$(echo "$out" | awk '/^DONE/{print $2}')
  echo "$(date '+%F %T') $NAME exists=$exists done=$done submits=$n" >> $LOG
  if [ "$exists" = 0 ] && [ -n "$done" ]; then
    if [ "$done" -gt 0 ]; then echo "$(date '+%F %T') $NAME has its test result" >> $LOG; exit 0; fi
    if [ "$n" -ge "$MAX" ]; then echo "$(date '+%F %T') $NAME: $MAX resubmits used, stopping" >> $LOG; exit 1; fi
    jid=$(timeout 60 ssh -o BatchMode=yes iitd-hpc08 "source /etc/profile; cd $P/code_lrc/scripts/hpc && qsub -N $NAME $*" 2>&1 | tail -1)
    n=$((n + 1))
    echo "$(date '+%F %T') resubmitted $NAME -> $jid" >> $LOG
  fi
  sleep 900
done
