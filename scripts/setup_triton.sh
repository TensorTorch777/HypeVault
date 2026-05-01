#!/usr/bin/env bash
# Prepare local Triton model repository layout (models populated by export_tensorrt.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT}/models/dinov2_classifier/1"
mkdir -p "${MODEL_DIR}"
echo "Model directory ready: ${MODEL_DIR}"
echo "Place model.onnx / TensorRT engine and config.pbtxt here after running scripts/export_tensorrt.py"
