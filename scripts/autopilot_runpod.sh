#!/usr/bin/env bash
# Full autopilot v2: handles HF download on pod + rsync from local + training + download
set -u

SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="root@154.54.102.35"
POD_PORT="10718"
SSH="ssh -i $SSH_KEY -p $POD_PORT -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh -i $SSH_KEY -p $POD_PORT -o ServerAliveInterval=30 -o StrictHostKeyChecking=no"
LOCAL_CKPT="/home/tensortorch26/Desktop/HypeVault/ml/checkpoints"
LOCAL_BEST="$LOCAL_CKPT/best_model.pt"
LOCAL_PARTIAL="$LOCAL_CKPT/.best_model.pt.qlZaoM"
LOG="$LOCAL_CKPT/autopilot.log"
SRC_ROOT="/home/tensortorch26/Desktop/scraper"
# Required: export HF_TOKEN='hf_...' before running (never commit tokens).
: "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face token (export HF_TOKEN=...)}"
EXPECTED_IMAGES=60000

mkdir -p "$LOCAL_CKPT"
exec > >(tee -a "$LOG") 2>&1
echo "=========================================="
echo "AUTOPILOT v2 STARTED: $(date)"
echo "=========================================="

count_pod_images() {
    $SSH "$POD_HOST" \
        "find /workspace/scraper -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \) 2>/dev/null | wc -l" 2>/dev/null || echo 0
}

count_pod_folder() {
    local folder="$1"
    $SSH "$POD_HOST" \
        "find /workspace/scraper/$folder -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \) 2>/dev/null | wc -l" 2>/dev/null || echo 0
}

ensure_hf_running() {
    # Check if HF fill is still running on pod
    local alive
    alive=$($SSH "$POD_HOST" "pgrep -f 'snapshot_download\|hf_fill' 2>/dev/null | head -1" 2>/dev/null || echo "")
    if [ -z "$alive" ]; then
        # Restart HF fill for remaining folders
        local need_restart=()
        for f in Label_0_Sneakers Label_0_Watches Label_1_Watches; do
            CNT=$(count_pod_folder "$f")
            if [ "$CNT" -lt 15000 ]; then
                need_restart+=("'$f'")
            fi
        done
        if [ ${#need_restart[@]} -gt 0 ]; then
            local folders_str
            folders_str=$(printf "%s," "${need_restart[@]}")
            folders_str="${folders_str%,}"
            echo "  HF fill not running, restarting for: $folders_str"
            $SSH "$POD_HOST" "nohup python3 -c \"
from huggingface_hub import snapshot_download
import os
for folder in [$folders_str]:
    cnt = len([f for f in __import__('pathlib').Path(f'/workspace/scraper/{folder}').rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]) if __import__('pathlib').Path(f'/workspace/scraper/{folder}').exists() else 0
    if cnt < 15000:
        print(f'HF downloading {folder} (have {cnt})...')
        snapshot_download(repo_id='TensorTorch777/sneakers-watches-dataset', repo_type='dataset', local_dir='/workspace/scraper', allow_patterns=[f'{folder}/**'], token='$HF_TOKEN', ignore_patterns=['*.metadata','.cache/**'])
        print(f'{folder} done')
print('HF fill complete')
\" >> /workspace/hf_fill.log 2>&1 &"
        fi
    fi
}

ensure_rsync_running() {
    # Keep Label_1_Sneakers syncing via rsync (already in progress)
    local cnt
    cnt=$(count_pod_folder "Label_1_Sneakers")
    local running
    running=$(pgrep -af "rsync.*Label_1_Sneakers" | grep -v pgrep | wc -l)
    if [ "$running" -eq 0 ] && [ "$cnt" -lt 15000 ]; then
        echo "  Restarting rsync for Label_1_Sneakers ($cnt/15000)"
        nohup rsync -av --partial --timeout=90 --no-owner --no-group --no-perms \
            -e "$RSYNC_SSH" \
            "$SRC_ROOT/Label_1_Sneakers/" \
            "$POD_HOST:/workspace/scraper/Label_1_Sneakers/" \
            >> "$LOCAL_CKPT/rsync2_Label_1_Sneakers.log" 2>&1 &
    fi
}

# ── PHASE 1: Wait for best_model.pt local download ─────────────────
echo "[PHASE 1] Waiting for best_model.pt to finish downloading locally..."
while true; do
    if [ -f "$LOCAL_BEST" ]; then
        SIZE=$(stat -c%s "$LOCAL_BEST" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt 4000000000 ]; then
            echo "  best_model.pt complete: $(ls -lh "$LOCAL_BEST" | awk '{print $5}') ✓"
            break
        fi
    fi
    PARTIAL_MB=0
    [ -f "$LOCAL_PARTIAL" ] && PARTIAL_MB=$(( $(stat -c%s "$LOCAL_PARTIAL" 2>/dev/null || echo 0) / 1024 / 1024 ))
    echo "  $(date -Is) best_model.pt: ${PARTIAL_MB} MB / ~4300 MB"
    sleep 30
done

# ── PHASE 2: Upload best_model.pt + history to pod ─────────────────
echo "[PHASE 2] Uploading best_model.pt to pod..."
$SSH "$POD_HOST" "mkdir -p /workspace/checkpoints"
rsync -av --partial --no-owner --no-group --no-perms \
    -e "$RSYNC_SSH" \
    "$LOCAL_BEST" \
    "$POD_HOST:/workspace/checkpoints/best_model.pt" && echo "  best_model.pt on pod ✓"
for f in history.json split_manifest.json; do
    [ -f "$LOCAL_CKPT/$f" ] && rsync -av --no-owner --no-group --no-perms \
        -e "$RSYNC_SSH" "$LOCAL_CKPT/$f" "$POD_HOST:/workspace/checkpoints/$f" 2>/dev/null || true
done
echo "  history + manifest uploaded ✓"

# ── PHASE 3: Monitor dataset until 60k images ──────────────────────
echo "[PHASE 3] Waiting for dataset to reach 60,000 images on pod..."
STALL_COUNT=0
PREV_TOTAL=0
while true; do
    TOTAL=$(count_pod_images)
    echo "  $(date -Is) pod images: $TOTAL / $EXPECTED_IMAGES"

    if [ "$TOTAL" -ge "$EXPECTED_IMAGES" ]; then
        echo "  Dataset complete: $TOTAL images ✓"
        break
    fi

    # Stall detection
    if [ "$TOTAL" -le "$PREV_TOTAL" ]; then
        STALL_COUNT=$(( STALL_COUNT + 1 ))
    else
        STALL_COUNT=0
    fi
    PREV_TOTAL="$TOTAL"

    if [ "$STALL_COUNT" -ge 3 ]; then
        echo "  Stall detected ($STALL_COUNT checks without progress) — restarting transfers"
        ensure_hf_running
        ensure_rsync_running
        STALL_COUNT=0
    else
        ensure_hf_running
        ensure_rsync_running
    fi

    sleep 120
done

# ── PHASE 4: Start training from epoch 2 ───────────────────────────
echo "[PHASE 4] Starting training (epochs 2-4, patience=2)..."
$SSH "$POD_HOST" "kill -9 \$(pgrep -f train.py) 2>/dev/null; rm -f /workspace/train.log; echo cleared"
sleep 3
$SSH "$POD_HOST" "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python3 /workspace/HypeVault/ml/train.py \
  --data_root /workspace/scraper \
  --output_dir /workspace/checkpoints \
  --model_name dinov2_vitg14_reg \
  --img_size 518 \
  --batch_size 8 \
  --accumulate_grad 4 \
  --num_epochs 3 \
  --start_epoch 2 \
  --resume_from /workspace/checkpoints/best_model.pt \
  --lr 1e-5 \
  --amp_dtype bf16 \
  --num_workers 8 \
  --save_every 1 \
  --early_stop_patience 2 \
  > /workspace/train.log 2>&1 & echo Training PID: \$!"
sleep 5
TPID=$($SSH "$POD_HOST" "pgrep -f train.py | head -1" 2>/dev/null || echo "")
echo "  Training PID on pod: $TPID ✓"

# ── PHASE 5: Monitor training ───────────────────────────────────────
echo "[PHASE 5] Monitoring training..."
TRAIN_STALL=0
PREV_LOG_SIZE=0
while true; do
    sleep 120
    ALIVE=$($SSH "$POD_HOST" "pgrep -f train.py | head -1" 2>/dev/null || echo "")
    LOG_LINES=$($SSH "$POD_HOST" "tail -c 800 /workspace/train.log 2>/dev/null" | \
        grep -aE 'Epoch|Val |Train |★|Early|complete|loss=' | tail -5 2>/dev/null || echo "")
    LOG_SIZE=$($SSH "$POD_HOST" "stat -c%s /workspace/train.log 2>/dev/null" || echo 0)
    GPU=$($SSH "$POD_HOST" "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader" 2>/dev/null || echo "N/A")
    HIST=$($SSH "$POD_HOST" "cat /workspace/checkpoints/history.json 2>/dev/null" | python3 -c "
import json,sys
try:
    h=json.load(sys.stdin)
    epochs=h.get('epochs',[])
    for e in epochs[-2:]:
        print(f\"  Epoch {e['epoch']}: val_acc={e['val_acc']*100:.2f}% val_f1={e['val_f1']:.4f}\")
except: pass
" 2>/dev/null || echo "")

    echo "--- $(date -Is) | alive=$ALIVE | gpu=$GPU ---"
    [ -n "$LOG_LINES" ] && echo "$LOG_LINES"
    [ -n "$HIST" ] && echo "$HIST"

    # Log stall detection
    if [ "$LOG_SIZE" -le "$PREV_LOG_SIZE" ] && [ -n "$ALIVE" ]; then
        TRAIN_STALL=$(( TRAIN_STALL + 1 ))
        if [ "$TRAIN_STALL" -ge 5 ]; then
            echo "  Training log stalled 5 checks — killing and restarting"
            $SSH "$POD_HOST" "kill -9 \$(pgrep -f train.py) 2>/dev/null; sleep 2; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python3 /workspace/HypeVault/ml/train.py --data_root /workspace/scraper --output_dir /workspace/checkpoints --model_name dinov2_vitg14_reg --img_size 518 --batch_size 8 --accumulate_grad 4 --num_epochs 3 --start_epoch 2 --resume_from /workspace/checkpoints/best_model.pt --lr 1e-5 --amp_dtype bf16 --num_workers 8 --save_every 1 --early_stop_patience 2 >> /workspace/train.log 2>&1 &" 2>/dev/null
            TRAIN_STALL=0
        fi
    else
        TRAIN_STALL=0
    fi
    PREV_LOG_SIZE="$LOG_SIZE"

    if [ -z "$ALIVE" ]; then
        echo "Training process ended."
        break
    fi
    if echo "$LOG_LINES" | grep -qE 'Training complete|Early stopping|Checkpoints in'; then
        echo "Training finished!"
        sleep 30
        break
    fi
done

# ── PHASE 6: Download all results ───────────────────────────────────
echo "[PHASE 6] Downloading results to local machine..."
rsync -av --partial --no-owner --no-group --no-perms \
    -e "$RSYNC_SSH" \
    "$POD_HOST:/workspace/checkpoints/best_model.pt" \
    "$LOCAL_CKPT/best_model_final.pt" && echo "  best_model_final.pt ✓"

for f in history.json split_manifest.json config.json training_curves.png confusion_matrix_eval.png; do
    rsync -av --no-owner --no-group --no-perms \
        -e "$RSYNC_SSH" \
        "$POD_HOST:/workspace/checkpoints/$f" \
        "$LOCAL_CKPT/$f" 2>/dev/null && echo "  $f ✓" || true
done
rsync -av --no-owner --no-group --no-perms \
    -e "$RSYNC_SSH" \
    "$POD_HOST:/workspace/train.log" \
    "$LOCAL_CKPT/train_final.log" 2>/dev/null && echo "  train_final.log ✓" || true

echo "=========================================="
echo "AUTOPILOT DONE: $(date)"
echo "*** TERMINATE THE RUNPOD POD NOW TO STOP BILLING ***"
echo "Saved to: $LOCAL_CKPT"
ls -lh "$LOCAL_CKPT/"*.pt "$LOCAL_CKPT/"*.png "$LOCAL_CKPT/"*.json 2>/dev/null | grep -v '^total'
echo "=========================================="
