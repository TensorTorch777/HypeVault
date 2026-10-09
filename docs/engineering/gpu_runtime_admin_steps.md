# GPU container runtime: steps for the machine administrator

These steps need root. They were not run by the agent. Source: [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), read on 2026-10-09.

Current state, from `docs/engineering/runtime_diagnostics_2026-10-09.txt`:
- Driver 580.178.04 is healthy. Do not change the driver.
- No `nvidia-container-*` packages are installed and there is no `/etc/docker/daemon.json`. There is no CDI spec in `/etc/cdi` or `/var/run/cdi`.
- `/` had 915 MB free at the diagnostic run and 42 GB later the same day.

## 1. Free disk space first

The pinned Triton image is 7.64 GB compressed and larger once extracted. Aim for at least 25 GB free before step 4. A later check on 2026-10-09 found 42 GB free, so this step may already be met; re-check with `df -h /` first.

Docker reports 0 B of reclaimable image space and 257 MB of reclaimable build cache. Freeing real space means deciding which files on `/` can go. Do not delete checkpoints under `ml_rtx5080/experiments/` without the owner's sign-off: they include the frozen DINOv3 checkpoint and the Phase 45 fold artifacts.

## 2. Install the NVIDIA Container Toolkit (Ubuntu 24.04, apt)

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1
sudo apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

## 3. Configure Docker

`/etc/docker/daemon.json` does not exist today, so there is nothing to back up. If it exists by the time you run this, copy it first.

```bash
[ -f /etc/docker/daemon.json ] && sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak-$(date +%F)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Restarting Docker stops running containers. The `fonoster-*` stack runs on this host, so schedule the restart.

## 4. Verify, then pull the pinned Triton image

```bash
docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.01-py3@sha256:c9f2ede50ccc4a3ce66e22e26ed1b5adc0c68516b58edfaca738693719c2146b nvidia-smi
```

Accept only if `nvidia-smi` inside the container lists the RTX 5080. Triton 26.01 documents driver 575 or later and Blackwell support, and ships ONNX Runtime 1.24.1, which reads IR 10.

## 5. Then the agent can continue

The selector models are `dinov2_vitb14_live` (the live ViT-B/14 at 504) and `dinov3_authenticity_candidate`. `dinov2_classifier` is not routed and stays untouched. The reviewed GPU configs are already tracked as `infra/triton/*/config.gpu.pbtxt`. Install them only after step 4 passes.

```bash
IMAGE=nvcr.io/nvidia/tritonserver:26.01-py3@sha256:c9f2ede50ccc4a3ce66e22e26ed1b5adc0c68516b58edfaca738693719c2146b
cp infra/triton/dinov2_vitb14_live/config.gpu.pbtxt models/dinov2_vitb14_live/config.pbtxt
cp infra/triton/dinov3_authenticity_candidate/config.gpu.pbtxt models/dinov3_authenticity_candidate/config.pbtxt
docker run -d --rm --name hv-triton-gpu --gpus all -p 18000:8000 -p 18001:8001 -p 18002:8002 \
  -v "$PWD/models:/models:ro" "$IMAGE" tritonserver --model-repository=/models \
  --model-control-mode=explicit --load-model=dinov2_vitb14_live --load-model=dinov3_authenticity_candidate
.venv/bin/python ml_rtx5080/dinov2_live_parity.py --stage triton --require-gpu
.venv/bin/python ml_rtx5080/triton_parity.py --require-gpu
curl -s localhost:18002/metrics | rg 'nv_gpu_(utilization|memory_used_bytes)'
```

Both parity protocols are already frozen:
- DINOv2: `dinov2_live_triton_gpu_protocol_v1.json`, SHA-256 `c4ec2851…e710`.
- DINOv3: `parity_protocol_v1.json`, SHA-256 `858abd0c…e4cb`.

A model becomes selectable on the GPU only in a reviewed commit that adds `KIND_GPU` to its `validated_instance_kinds` in `backend/inference/model_router.py` and cites the passing result. Restore the CPU configs from `infra/triton/*/config.pbtxt` to go back.

## Stop conditions

- Any step asks to replace or downgrade the NVIDIA driver.
- The container still cannot see the GPU after step 3.
- ONNX Runtime falls back to CPU for the CUDA provider on the RTX 5080. Early-2026 public reports say prebuilt CUDA packages lacked sm_120 kernels, so GPU serving must not be claimed in this case.
