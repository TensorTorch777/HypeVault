# HypeVault — Model Training Guide

DINOv2-Giant (1.1B params) full end-to-end fine-tuning on  
**RTX 6000 Pro Blackwell (96 GB VRAM)** — cloud GPU.

---

## Dataset

| Folder | Label | Meaning | Count |
|---|---|---|---|
| `Label_0_Sneakers` | **0** | Authentic | 15,000 |
| `Label_0_Watches` | **0** | Authentic | 15,000 |
| `Label_1_Sneakers` | **1** | Deepfake | 15,000 |
| `Label_1_Watches` | **1** | Deepfake | 15,000 |
| **Total** | | | **60,000** |

---

## Step-by-Step on the Cloud GPU

### 1 — Upload the dataset

```bash
# Zip on your local machine:
zip -r scraper_dataset.zip /home/tensortorch26/Desktop/scraper/

# Upload to cloud instance (replace with actual IP):
scp scraper_dataset.zip user@<GPU_IP>:~/
ssh user@<GPU_IP>
unzip scraper_dataset.zip -d ~/scraper
```

### 2 — Set up Python environment

```bash
# Python 3.11+ recommended
python3 -m venv ~/hypevault_env
source ~/hypevault_env/bin/activate

# Install PyTorch with CUDA 12.4 (Blackwell: CUDA 12.4+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install rest of deps
cd ~/HypeVault/ml
pip install -r requirements_train.txt
```

### 3 — Upload the project

```bash
# From local machine:
scp -r /home/tensortorch26/Desktop/HypeVault/ml user@<GPU_IP>:~/HypeVault/ml
```

### 4 — Run training

#### Option A — Budget mode ($4–5, fits $6.49 balance) ← START HERE

Two-stage progressive fine-tuning:

```bash
source ~/hypevault_env/bin/activate
cd ~/HypeVault/ml

python train_staged.py \
  --data_root ~/scraper \
  --output_dir ~/HypeVault/ml/checkpoints \
  --model_name dinov2_vitg14_reg \
  --img_size 518 \
  --amp_dtype bf16 \
  --num_workers 8 \
  --stage both
```

| Stage | What trains | Time | Cost | Expected acc |
|---|---|---|---|---|
| 1 | Head only (frozen backbone) | ~7 min | ~$0.22 | ~80–88% |
| 2 | Last 8 blocks + head | ~2 hrs | ~$3.90 | ~91–95% |
| **Total** | | **~2.2 hrs** | **~$4.12** | **~93%** |

#### Option B — Full fine-tune ($7–12, top up balance later)

```bash
python train.py \
  --data_root ~/scraper \
  --output_dir ~/HypeVault/ml/checkpoints \
  --model_name dinov2_vitg14_reg \
  --img_size 518 \
  --batch_size 48 \
  --num_epochs 20 \
  --lr 1e-5 \
  --amp_dtype bf16 \
  --num_workers 8
```

For products held out from training (recommended before production accuracy claims):

```bash
python train.py \
  --data_root ~/scraper \
  --test_root ~/scraper_held_out \
  --output_dir ~/HypeVault/ml/checkpoints \
  --model_name dinov2_vitg14_reg \
  --img_size 518 \
  --batch_size 48 \
  --num_epochs 20 \
  --lr 1e-5 \
  --amp_dtype bf16 \
  --num_workers 8
```

`train.py` splits by product folder (brand/model directory), not by individual image, so near-duplicate listing photos from the same SKU do not leak across train and validation.

Expected training time: **~4–6 hours**, ~**$8–11**. Use after topping up.

### 5 — Monitor training

```bash
# In another terminal:
watch -n 30 "tail -30 ~/HypeVault/ml/checkpoints/history.json"

# Or use htop + nvidia-smi
nvidia-smi dmon -s u -d 5
```

### 6 — Export ONNX → TensorRT FP16

```bash
python export_tensorrt.py \
  --onnx  checkpoints/dinov2_hypevault.onnx \
  --engine checkpoints/dinov2_hypevault_fp16.trt \
  --img_size 518 \
  --max_batch 32 \
  --verify
```

TRT build takes ~5–15 minutes. Output: `dinov2_hypevault_fp16.trt`

### 7 — Deploy to Triton (production)

Copy engine to Triton model repository:

```bash
MODEL_REPO=/opt/triton/model_repository
mkdir -p $MODEL_REPO/hypevault_dinov2/1

cp checkpoints/dinov2_hypevault_fp16.trt \
   $MODEL_REPO/hypevault_dinov2/1/model.plan

# Write config.pbtxt
cat > $MODEL_REPO/hypevault_dinov2/config.pbtxt << 'EOF'
name: "hypevault_dinov2"
backend: "tensorrt"
max_batch_size: 32

input [{ name: "input"  data_type: TYPE_FP32  dims: [3, 518, 518] }]
output[{ name: "logit"  data_type: TYPE_FP32  dims: [1] }]

dynamic_batching {
  preferred_batch_size: [1, 4, 8, 16, 32]
  max_queue_delay_microseconds: 2000
}
EOF

# Start Triton
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $MODEL_REPO:/models \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
```

### 8 — Download weights back to local machine

```bash
# From local:
scp user@<GPU_IP>:~/HypeVault/ml/checkpoints/best_model.pt \
    /home/tensortorch26/Desktop/HypeVault/ml/checkpoints/
scp user@<GPU_IP>:~/HypeVault/ml/checkpoints/dinov2_hypevault_fp16.trt \
    /home/tensortorch26/Desktop/HypeVault/ml/checkpoints/
```

---

## Key Architecture Decisions

| Setting | Value | Why |
|---|---|---|
| Backbone | DINOv2-Giant | 1.1B params, understands 3D geometry (not just pixel texture) |
| Input resolution | **518 × 518** | Native DINOv2 patch grid (14×14 px patches → 37×37=1369 patches) |
| Training mode | **Full fine-tune** | All 1.1B params trained end-to-end — needs 96 GB VRAM |
| Loss | BCEWithLogitsLoss | Binary task (AUTHENTIC vs DEEPFAKE) |
| Optimizer | AdamW | β=(0.9, 0.999), ε=1e-8 |
| LR | 1e-5 backbone, 1e-4 head | Layer-wise decay — backbone needs slow updates |
| Scheduler | Cosine annealing + 2-epoch warmup | Standard for ViT fine-tuning |
| Precision | **BF16** | Native on Blackwell/Ampere+, no loss scaling needed |
| Augmentation | RandomResizedCrop, ColorJitter, Mixup α=0.2 | Improve generalisation to unseen deepfakes |
| Weighted sampler | ✓ | Ensures 50/50 authentic/fake per batch |

---

## Validation vs held-out test

Validation accuracy is measured on products from the **same scraped distribution** as training (StockX-style folders), with splits at the **product** level so duplicate angles of one listing do not appear in both train and val.

That same-distribution validation can still look strong when val products resemble train products (brand, lighting, marketplace). It is **not** a substitute for generalization to unseen products or marketplaces.

For production accuracy claims, reserve a **held-out test root** of products never used in training and pass `--test_root`. `history.json` stores per-epoch validation under `epochs` and final held-out metrics under `test` when `--test_root` is set.

---

## Expected Results (from PPT literature)

Based on similar DINOv2 binary classification work  
(Garcia-Cotte et al. 2024, arXiv:2410.05969):

| Metric | Same-distribution val | Held-out test |
|---|---|---|
| Accuracy | **>95%** (typical) | **Lower** — use for production claims |
| F1 | >0.95 (typical) | Report separately from val |
| Inference latency (TRT FP16) | **<200ms** | Same deployment path |

Treat validation as a training monitor; treat held-out test as the bar for deployment expectations.

---

## Files

```
ml/
├── train.py              ← main training script
├── export_tensorrt.py    ← ONNX → TRT FP16 export
├── requirements_train.txt
├── README_training.md    ← this file
└── checkpoints/          ← created at runtime
    ├── best_model.pt
    ├── final_model.pt
    ├── dinov2_hypevault.onnx
    ├── dinov2_hypevault_fp16.trt
    ├── config.json
    └── history.json
```
