#!/bin/bash
# One-shot: wait for the SPEED+ image upload (rsync PID $1) to finish, check the file count on the cluster,
# submit the SPEED+ LiePose job and move it ahead of wmd-energy in our queue order (qorder swaps two jobs).
P=/scratch/hpc/visitor/dtu001.visitor/liepose
LOG=/media/kavinder/hdd2/liepose_data/hpc/submit_speedplus.log
while kill -0 $1 2>/dev/null; do sleep 60; done
n=$(timeout 60 ssh -o BatchMode=yes iitd-hpc08 "ls $P/data/speedplusv2/synthetic/images | wc -l")
echo "$(date '+%F %T') upload done, $n images on cluster" >> $LOG
if [ "$n" != "59960" ]; then echo "$(date '+%F %T') image count wrong, not submitting" >> $LOG; exit 1; fi
out=$(timeout 120 ssh -o BatchMode=yes iitd-hpc08 "cd $P/code && qsub -N sp-se3 scripts/hpc/speedplus_liepose_scai.pbs" 2>&1 | tail -1)
echo "$(date '+%F %T') qsub sp-se3: $out" >> $LOG
case "$out" in *pbshpc*) ;; *) exit 1 ;; esac
en=$(timeout 60 ssh -o BatchMode=yes iitd-hpc08 "qselect -u dtu001.visitor -N wmd-energy -s Q" | head -1)
if [ -n "$en" ]; then
  r=$(timeout 60 ssh -o BatchMode=yes iitd-hpc08 "qorder ${out%%.*} ${en%%.*} 2>&1; qstat -u dtu001.visitor | grep -E 'sp-se3|wmd'")
  echo "$(date '+%F %T') qorder sp-se3 <-> wmd-energy: $r" >> $LOG
fi
