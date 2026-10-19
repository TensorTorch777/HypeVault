#!/usr/bin/env bash
# Wait until no training job is using the GPU, then generate backdrops and final renders.
#   render/gpu_pipeline.sh            wait, then run everything
#   NO_WAIT=1 render/gpu_pipeline.sh  run now
#   WAIT_PID=<pid> render/gpu_pipeline.sh  also wait for that process (e.g. a training sweep driver) to exit
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/.."

if [ -z "${NO_WAIT:-}" ]; then
  echo "[gpu] waiting for training to finish ($(date))"
  # WAIT_PID: the sweep driver, so a gap between two training runs is not mistaken for "done".
  while { [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; } \
     || pgrep -f "ml_rtx5080/(train|stage1_compare|evaluate)\.py" >/dev/null \
     || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 5000 ]; do
    sleep 120
  done
  echo "[gpu] GPU is free ($(date))"
fi

echo "[gpu] SDXL backdrops"
"$REPO/.venv/bin/python" "$HERE/backdrops.py" --variants 2 || echo "[gpu] backdrops failed"

echo "[gpu] final Cycles renders"
bash "$HERE/render_all.sh" final || echo "[gpu] renders failed"
echo "[gpu] done ($(date))"
