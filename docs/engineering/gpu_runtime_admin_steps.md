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

1. Start Triton with `--gpus all`, explicit model control, and only `dinov2_classifier` and `dinov3_authenticity_candidate`.
2. Change `instance_group` to `KIND_GPU` only in a new reviewed config, and only after step 4 passes.
3. Run `python ml_rtx5080/triton_parity.py --require-gpu`. The protocol and tolerances are already frozen.
4. Confirm GPU execution in Triton metrics (`nv_gpu_utilization`, `nv_gpu_memory_used_bytes`) during inference.

## Stop conditions

- Any step asks to replace or downgrade the NVIDIA driver.
- The container still cannot see the GPU after step 3.
- ONNX Runtime falls back to CPU for the CUDA provider on the RTX 5080. Early-2026 public reports say prebuilt CUDA packages lacked sm_120 kernels, so GPU serving must not be claimed in this case.
