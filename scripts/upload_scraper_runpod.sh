#!/usr/bin/env bash
set -u

SRC_ROOT="/home/tensortorch26/Desktop/scraper"
DST_HOST="root@154.54.102.53"
DST_ROOT="/workspace/scraper"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_PORT="18332"
LOG_FILE="/home/tensortorch26/Desktop/HypeVault/ml/checkpoints/scraper_upload.log"
STALL_SECONDS=180
RSYNC_TIMEOUT=120

FOLDERS=(
  "Label_0_Sneakers"
  "Label_0_Watches"
  "Label_1_Sneakers"
  "Label_1_Watches"
)

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

ssh_base=(ssh -i "$SSH_KEY" -p "$SSH_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
rsync_ssh="ssh -i $SSH_KEY -p $SSH_PORT -o ServerAliveInterval=30 -o ServerAliveCountMax=10"

echo "=== scraper upload started $(date -Is) ==="
"${ssh_base[@]}" "$DST_HOST" "mkdir -p $DST_ROOT"

count_images() {
  local root="$1"
  find "$root" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) 2>/dev/null | wc -l
}

count_remote_brand() {
  local folder="$1"
  local brand="$2"
  "${ssh_base[@]}" "$DST_HOST" \
    "find '$DST_ROOT/$folder/$brand' -type f \\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \\) 2>/dev/null | wc -l"
}

list_brands() {
  local folder="$1"
  find "$SRC_ROOT/$folder" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort
}

sync_brand() {
  local folder="$1"
  local brand="$2"
  local expected
  local attempt=1

  expected="$(count_images "$SRC_ROOT/$folder/$brand")"
  while true; do
    echo "--- [$folder/$brand] attempt $attempt expected=$expected $(date -Is) ---"
    local before
    before="$(count_remote_brand "$folder" "$brand")"
    if [ "$before" -eq "$expected" ]; then
      echo "[$folder/$brand] complete"
      return 0
    fi

    rsync -av --partial --timeout="$RSYNC_TIMEOUT" --no-owner --no-group --no-perms --info=progress2 \
      -e "$rsync_ssh" \
      "$SRC_ROOT/$folder/$brand/" \
      "$DST_HOST:$DST_ROOT/$folder/$brand/" &
    local rsync_pid=$!
    local idle=0

    while kill -0 "$rsync_pid" 2>/dev/null; do
      sleep 30
      local current
      current="$(count_remote_brand "$folder" "$brand")"
      if [ "$current" -le "$before" ]; then
        idle=$((idle + 30))
      else
        before="$current"
        idle=0
      fi
      if [ "$idle" -ge "$STALL_SECONDS" ]; then
        echo "[$folder/$brand] stalled for ${STALL_SECONDS}s at $current/$expected, restarting rsync"
        kill "$rsync_pid" 2>/dev/null || true
        wait "$rsync_pid" 2>/dev/null || true
        break
      fi
    done

    if kill -0 "$rsync_pid" 2>/dev/null; then
      wait "$rsync_pid" 2>/dev/null || true
    else
      wait "$rsync_pid" 2>/dev/null || true
    fi

    local after
    after="$(count_remote_brand "$folder" "$brand")"
    echo "[$folder/$brand] remote image count: $after"
    if [ "$after" -eq "$expected" ]; then
      echo "[$folder/$brand] complete"
      return 0
    fi

    attempt=$((attempt + 1))
    sleep 5
  done
}

for folder in "${FOLDERS[@]}"; do
  while IFS= read -r brand; do
    [ -n "$brand" ] || continue
    sync_brand "$folder" "$brand"
  done < <(list_brands "$folder")
done

total="$("${ssh_base[@]}" "$DST_HOST" \
  "find '$DST_ROOT' -type f \\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \\) | wc -l")"
size="$("${ssh_base[@]}" "$DST_HOST" "du -sh '$DST_ROOT' | awk '{print \$1}'")"
echo "=== scraper upload finished $(date -Is) ==="
echo "remote total images: $total"
echo "remote total size: $size"
