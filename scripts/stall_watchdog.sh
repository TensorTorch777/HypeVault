#!/usr/bin/env bash
# Tight stall watchdog for Label_0 rsyncs
# Run: nohup ./stall_watchdog.sh &
KEY=~/.ssh/id_ed25519
POD=root@154.54.102.35
PORT=10718
SRC=/home/tensortorch26/Desktop/scraper
CKPT=/home/tensortorch26/Desktop/HypeVault/ml/checkpoints
LOG=$CKPT/watchdog.log
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"

pod_count() {
  ssh -i $KEY -p $PORT $SSH_OPTS $POD \
    "find /workspace/scraper -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \) 2>/dev/null | wc -l" 2>/dev/null || echo 0
}

restart_rsyncs() {
  pkill -f 'rsync.*Label_0' 2>/dev/null; sleep 2
  for folder in Label_0_Sneakers Label_0_Watches; do
    CNT=$(ssh -i $KEY -p $PORT $SSH_OPTS $POD "find /workspace/scraper/$folder -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \) 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    if [ "$CNT" -lt 15000 ]; then
      nohup rsync -av --partial --timeout=90 --no-owner --no-group --no-perms \
        -e "ssh -i $KEY -p $PORT -o ServerAliveInterval=30 -o StrictHostKeyChecking=no" \
        "$SRC/$folder/" "$POD:/workspace/scraper/$folder/" \
        >> "$CKPT/rsync3_${folder}.log" 2>&1 &
      echo "$(date -Is) restarted $folder ($CNT/15000) PID=$!" >> $LOG
    fi
  done
}

echo "$(date -Is) watchdog started" >> $LOG
prev=0; stall=0
while true; do
  sleep 60
  total=$(pod_count)
  echo "$(date -Is) total=$total stall=$stall" >> $LOG
  if [ "$total" -ge 60000 ]; then echo "$(date -Is) DONE 60000" >> $LOG; exit 0; fi
  if [ "$total" -le "$prev" ]; then
    stall=$((stall+1))
    if [ "$stall" -ge 2 ]; then
      echo "$(date -Is) STALL - restarting" >> $LOG
      restart_rsyncs
      stall=0
    fi
  else
    stall=0
  fi
  prev=$total
done
