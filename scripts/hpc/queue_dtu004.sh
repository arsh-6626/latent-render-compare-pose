#!/bin/bash
# dtu004 queue keeper (runs locally): keeps the listed jobs alive under scai_q's 2-jobs-per-account limit. Every 15 min:
# a job whose name is not in the queue and whose run has no test result yet is (re)submitted when a slot is free;
# runs resume from their latest checkpoint. Stops when every listed run has its esa.json, or after MAX submits each.
#   queue_dtu004.sh   (job list below)
P=/scratch/hpc/visitor/dtu001.visitor/liepose; W=/scratch/hpc/visitor/dtu004.visitor/larp
LOG=/media/kavinder/hdd2/liepose_data/hpc/queue_dtu004.log; MAX=4
JOBS=("fd400-nowm|MODE=field_nowm,STEPS=400000,BS=64,NS=64,TAG=_400k|train_steps=400000+seed=42+den_mode=field_nowm+field=sp_field_v1"
      "fd400-lat|MODE=field_full_lat,STEPS=400000,BS=64,NS=64,TAG=_400k|train_steps=400000+seed=42+den_mode=field_full_lat+field=sp_field_v1")
declare -A n
while true; do
  st=$(timeout 120 ssh -o BatchMode=yes yashdeep@10.141.90.34 "ssh iitd-hpc 'source /etc/profile; qstat -u dtu004.visitor 2>/dev/null | tail -n +6 | awk \"{print \\\$4}\"; for e in $W/experiments/speedplus-field/*; do ls \$e/test_*/esa.json >/dev/null 2>&1 && echo DONE \$(basename \$e); done'" 2>/dev/null)
  [ -z "$st" ] && { echo "$(date '+%F %T') unreachable" >> $LOG; sleep 900; continue; }
  used=$(echo "$st" | grep -vc "^DONE")
  all_done=1
  for j in "${JOBS[@]}"; do
    IFS='|' read name vars exp <<< "$j"
    if echo "$st" | grep -q "^DONE $exp$"; then continue; fi
    all_done=0
    echo "$st" | grep -qx "$name" && continue
    if [ "$used" -lt 2 ] && [ "${n[$name]:-0}" -lt $MAX ]; then
      r=$(timeout 60 ssh -o BatchMode=yes yashdeep@10.141.90.34 "ssh iitd-hpc 'source /etc/profile; cd $P/code_larp/scripts/hpc; qsub -N $name -v $vars field_dit_dtu004.pbs'" 2>&1 | tail -1)
      n[$name]=$(( ${n[$name]:-0} + 1 )); used=$((used + 1))
      echo "$(date '+%F %T') submitted $name ($vars) -> $r [submit ${n[$name]}]" >> $LOG
    fi
  done
  echo "$(date '+%F %T') poll: used=$used" >> $LOG
  [ "$all_done" = 1 ] && { echo "$(date '+%F %T') all runs have test results" >> $LOG; exit 0; }
  sleep 900
done
