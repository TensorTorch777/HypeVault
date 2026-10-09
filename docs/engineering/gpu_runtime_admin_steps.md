# GPU container runtime: steps for the machine administrator

**Status, 2026-10-09 17:06: done.** The owner ran `sudo bash scripts/install_nvidia_container_toolkit.sh` (local, gitignored). Toolkit 1.20.1 is installed, the CDI spec lists `nvidia.com/gpu=all`, Docker has the `nvidia` runtime, and the pinned image sees the RTX 5080. Results are in `ml_rtx5080/experiments/dual_model_triton_v1/phase51_report.md`.

Use the exact-FP32 configs `infra/triton/*/config.gpu_fp32.pbtxt` for serving, not `config.gpu.pbtxt`, which failed parity under TF32. The Docker restart left 4 `fonoster` containers stopped because their project files were missing (report section 7).

These steps need root. The agent cannot run them because `sudo` requires a password. The agent will not ask for, store, or work around that password.

Sources, read on 2026-10-09:
- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Support for Container Device Interface](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)

## Current state

Re-checked for Phase 51; full output is in `docs/engineering/runtime_diagnostics_2026-10-09_phase51.txt`.

| Item | State |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS (noble), kernel 7.0.0-38-generic |
| Driver | 580.178.04, healthy. Do not change it |
| Docker | 29.4.3; runtimes `runc` only; CDI spec dirs `/etc/cdi`, `/var/run/cdi` (neither exists) |
| Toolkit | Not installed. No `nvidia-ctk`, no `nvidia-container-*` or `libnvidia-container*` packages, no `/etc/docker/daemon.json` |
| Failure | `docker run --gpus all …` → `failed to discover GPU vendor from CDI: no known GPU vendor found`. Docker 29 resolves `--gpus` through CDI, and there is no NVIDIA CDI spec |
| Pinned image | `nvcr.io/nvidia/tritonserver:26.01-py3@sha256:c9f2ede50ccc4a3ce66e22e26ed1b5adc0c68516b58edfaca738693719c2146b` is already pulled (Triton 2.65.0, CUDA 13.1.1) |
| Disk | 23 GB free on `/` |

## 1. Install the NVIDIA Container Toolkit (Ubuntu, apt)

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
nvidia-ctk --version
```

These packages do not touch the driver. If apt proposes installing, removing, or upgrading any `nvidia-driver-*`, `libnvidia-compute-*`, or kernel module package, stop and do not confirm.

## 2. Confirm the CDI specification

Toolkit 1.18 and later generate `/var/run/cdi/nvidia.yaml` through the `nvidia-cdi-refresh` service.

```bash
systemctl status nvidia-cdi-refresh.path nvidia-cdi-refresh.service --no-pager
nvidia-ctk cdi list
```

`nvidia-ctk cdi list` should include `nvidia.com/gpu=0` (or the GPU UUID) and `nvidia.com/gpu=all`. If it lists nothing, regenerate the spec:

```bash
sudo systemctl restart nvidia-cdi-refresh.service || sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
nvidia-ctk --debug cdi list
```

## 3. Configure Docker and restart it

`/etc/docker/daemon.json` does not exist today. If it exists by the time you run this, back it up first.

```bash
[ -f /etc/docker/daemon.json ] && sudo cp -a /etc/docker/daemon.json /etc/docker/daemon.json.bak-$(date +%F-%H%M)
sudo nvidia-ctk runtime configure --runtime=docker
cat /etc/docker/daemon.json
```

The configure command adds an `nvidia` runtime entry and keeps unrelated settings.

**Restarting Docker interrupts running containers.** Eleven `fonoster-*` containers are running on this host (apiserver, asterisk, routr, postgres, influxdb, autopilot, nats, autoheal, dashboard, envoy, rtpengine). Restart only when their owner agrees to the interruption.

```bash
docker ps --format '{{.Names}} {{.Status}}'
sudo systemctl restart docker
docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}'
```

The `fonoster` containers come back only if their restart policy allows it. Check `docker ps` afterwards.

## 4. Verify GPU access inside a container

```bash
IMAGE=nvcr.io/nvidia/tritonserver:26.01-py3@sha256:c9f2ede50ccc4a3ce66e22e26ed1b5adc0c68516b58edfaca738693719c2146b
docker run --rm --gpus all "$IMAGE" nvidia-smi
docker run --rm --gpus all "$IMAGE" nvidia-smi -L
docker run --rm --device nvidia.com/gpu=all "$IMAGE" nvidia-smi -L
```

Accept only if the container lists `NVIDIA GeForce RTX 5080` with driver 580.178.04. Then tell the agent, which will run the step 5 checks and a CUDA operation inside the container.

## 5. Agent runbook after step 4 passes

The selector models are `dinov2_vitb14_live` (the live ViT-B/14 at 504) and `dinov3_authenticity_candidate`. `dinov2_classifier` is not routed and stays untouched. The reviewed GPU configs are tracked as `infra/triton/*/config.gpu.pbtxt`.

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

Both parity protocols are frozen:
- DINOv2 GPU stage: `dinov2_live_triton_gpu_protocol_v1.json`, SHA-256 `c4ec2851…e710`.
- DINOv3: `parity_protocol_v1.json`, SHA-256 `858abd0c…e4cb`.

A model becomes selectable on the GPU only in a reviewed commit that adds `KIND_GPU` to its `validated_instance_kinds` in `backend/inference/model_router.py` and cites the passing result. To go back, restore the CPU configs from `infra/triton/*/config.pbtxt`.

## Stop conditions

- Any step asks to replace, upgrade, or downgrade the NVIDIA driver.
- The container still cannot see the GPU after step 3.
- ONNX Runtime falls back to CPU for the CUDA provider on the RTX 5080. Early-2026 public reports say prebuilt CUDA packages lacked sm_120 kernels, so GPU serving must be proven with Triton GPU metrics, not assumed.
