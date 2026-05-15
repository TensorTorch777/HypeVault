#!/usr/bin/env bash
set -u

SRC_ROOT="/home/tensortorch26/Desktop/scraper"
DST_HOST="root@154.54.102.53"
DST_ROOT="/workspace/scraper"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_PORT="18332"
LOG_DIR="/home/tensortorch26/Desktop/HypeVault/ml/checkpoints"
RSYNC_TIMEOUT=120

FOLDERS=(
  "Label_0_Sneakers"
  "Label_0_Watches"
  "Label_1_Sneakers"
  "Label_1_Watches"
)

mkdir -p "$LOG_DIR"
rsync_ssh="ssh -i $SSH_KEY -p $SSH_PORT -o ServerAliveInterval=30 -o ServerAliveCountMax=10"

echo "=== fast parallel rsync started $(date -Is) ==="

ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "$DST_HOST" \
  "pkill -f 'hf download' 2>/dev/null || true; mkdir -p $DST_ROOT"

count_remote() {
  local folder="$1"
  ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "$DST_HOST" \
    "find '$DST_ROOT/$folder' -type f \\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \\) 2>/dev/null | wc -l"
}

sync_folder() {
  local folder="$1"
  local expected
  local log_file="$LOG_DIR/rsync_${folder}.log"
  expected="$(find "$SRC_ROOT/$folder" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l)"

  while true; do
    echo "[$(date -Is)] $folder sync start (expected $expected)" >> "$log_file"
    if rsync -av --partial --timeout="$RSYNC_TIMEOUT" --no-owner --no-group --no-perms \
      -e "$rsync_ssh" \
      "$SRC_ROOT/$folder/" \
      "$DST_HOST:$DST_ROOT/$folder/" >> "$log_file" 2>&1; then
      local count
      count="$(count_remote "$folder")"
      echo "[$(date -Is)] $folder remote count $count" >> "$log_file"
      if [ "$count" -eq "$expected" ]; then
        echo "[$(date -Is)] $folder complete" >> "$log_file"
        return 0
      fi
    fi
    echo "[$(date -Is)] $folder retry in 10s" >> "$log_file"
    sleep 10
  done
}

pids=()
for folder in "${FOLDERS[@]}"; do
  sync_folder "$folder" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

total="$(ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "$DST_HOST" \
  "find '$DST_ROOT' -type f \\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \\) | wc -l")"
size="$(ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=10 "$DST_HOST" \
  "du -sh '$DST_ROOT' | awk '{print \$1}'")"
echo "=== fast parallel rsync finished $(date -Is) total_images=$total size=$size ==="
