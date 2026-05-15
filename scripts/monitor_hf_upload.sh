#!/usr/bin/env bash
set -u

LOG_FILE="/home/tensortorch26/Desktop/HypeVault/ml/checkpoints/hf_upload.log"
MONITOR_LOG="/home/tensortorch26/Desktop/HypeVault/ml/checkpoints/hf_upload_monitor.log"
PID_FILE="/home/tensortorch26/Desktop/HypeVault/ml/checkpoints/hf_upload_monitor.pid"

mkdir -p "$(dirname "$MONITOR_LOG")"
echo "$$" > "$PID_FILE"

echo "=== hf upload monitor started $(date -Is) ===" >> "$MONITOR_LOG"

while true; do
  if ! pgrep -f 'upload_dataset_hf_runpod.py' >/dev/null; then
    {
      echo "--- $(date -Is) upload process not running ---"
      tail -n 8 "$LOG_FILE" 2>/dev/null || true
    } >> "$MONITOR_LOG"
    echo "=== hf upload monitor finished $(date -Is) ===" >> "$MONITOR_LOG"
    rm -f "$PID_FILE"
    exit 0
  fi

  snapshot="$(grep -E '^Files:   hashed' "$LOG_FILE" 2>/dev/null | tail -n 1 || true)"
  batch="$(grep -E 'Processing Files \(' "$LOG_FILE" 2>/dev/null | tail -n 1 | tr -d '\r' || true)"
  {
    echo "--- $(date -Is) ---"
    if [ -n "$snapshot" ]; then
      echo "$snapshot"
    else
      echo "no snapshot line yet"
    fi
    if [ -n "$batch" ]; then
      echo "$batch"
    fi
  } >> "$MONITOR_LOG"

  sleep 60
done
