#!/bin/bash
# Keep a resumable job going in 24 h slots on either account (resume_loop.sh, generalised). Optionally first stops a
# running job right after its next checkpoint, so no training is lost when moving it to the shorter walltime.
# Whenever no job named NAME exists and EXP has no DONE file yet, resubmit (training resumes from the latest
# checkpoint; a finished run only tests). Runs locally, under nohup.
#   resume_loop_any.sh <dtu001|dtu004> <name> <exp-dir> <done-glob> <max-submits> <script-dir> <stop-jobid|-> <qsub args...>
ACCT=$1; NAME=$2; EXP=$3; DONE=$4; MAX=$5; SDIR=$6; STOP=$7; shift 7
LOG=/media/kavinder/hdd2/liepose_data/hpc/resume_loop_$NAME.log
rsh() {  # run a command on the account's login node (base64: no nested quoting)
  local c; c=$(printf '%s' "source /etc/profile; $1" | base64 -w0)
  if [ "$ACCT" = dtu001 ]; then timeout 150 ssh -o BatchMode=yes -o ConnectTimeout=30 iitd-hpc08 "echo $c | base64 -d | bash" 2>/dev/null
  else timeout 150 ssh -o BatchMode=yes yashdeep@10.141.90.34 "ssh iitd-hpc 'echo $c | base64 -d | bash'" 2>/dev/null; fi
}
latest() { rsh "ls -d '$EXP'/ckpt/checkpoint_* 2>/dev/null | sed 's/.*checkpoint_//' | sort -n | tail -1"; }
if [ "$STOP" != - ]; then
  c0=$(latest); c0=${c0:-0}
  echo "$(date '+%F %T') $NAME: stopping $STOP after the checkpoint following ${c0}" >> $LOG
  while true; do
    c=$(latest)
    if [ -n "$c" ] && [ "$c" -gt "$c0" ]; then
      sleep 180  # let the checkpoint finish writing
      echo "$(date '+%F %T') $NAME: checkpoint_$c written, qdel $STOP -> $(rsh "qdel $STOP && echo ok")" >> $LOG
      break
    fi
    sleep 300
  done
fi
n=0
while true; do
  out=$(rsh "echo EXISTS \$(qselect -u $ACCT.visitor -N $NAME 2>/dev/null | wc -l); ls $EXP/$DONE 2>/dev/null | wc -l | sed 's/^/DONE /'")
  exists=$(echo "$out" | awk '/^EXISTS/{print $2}'); done_=$(echo "$out" | awk '/^DONE/{print $2}')
  echo "$(date '+%F %T') $NAME exists=$exists done=$done_ submits=$n" >> $LOG
  if [ "$exists" = 0 ] && [ -n "$done_" ]; then
    if [ "$done_" -gt 0 ]; then echo "$(date '+%F %T') $NAME has its test result" >> $LOG; exit 0; fi
    if [ "$n" -ge "$MAX" ]; then echo "$(date '+%F %T') $NAME: $MAX resubmits used, stopping" >> $LOG; exit 1; fi
    jid=$(rsh "cd $SDIR && qsub -N $NAME $*" | tail -1)
    n=$((n + 1))
    echo "$(date '+%F %T') resubmitted $NAME -> $jid" >> $LOG
  fi
  sleep 900
done
