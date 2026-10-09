# Phase 51 report — NVIDIA Docker GPU access and Triton validation

GPU access inside Docker is still blocked. The NVIDIA Container Toolkit is not installed, so Docker 29 has no NVIDIA CDI device to inject, and installing it needs root (`sudo` requires a password this session does not have). Every GPU step stopped at that point. Nothing ran on the GPU, and no GPU result is claimed.

One question did get answered on CPU: the pinned Triton 26.01 image loads the live DINOv2 export (ONNX IR 10), which Triton 23.10 could not.

## Flags

`NVIDIA_TOOLKIT_INSTALLED = false`

`DOCKER_GPU_ACCESS = BLOCKED`

`RTX5080_VISIBLE_IN_CONTAINER = false`

`TRITON_VERSION = 2.65.0 (NGC 26.01, CUDA 13.1.1, image sha256:c9f2ede50ccc4a3ce66e22e26ed1b5adc0c68516b58edfaca738693719c2146b); CPU only`

`DINOV2_TRITON_READY = false` (loads and passes parity on CPU Triton; GPU blocked; selector keeps it unavailable)

`DINOV3_TRITON_READY = false` (ready on CPU Triton only; GPU blocked)

`DINOV2_GPU_PARITY = BLOCKED`

`DINOV3_GPU_PARITY = BLOCKED`

`GPU_PERFORMANCE_MEASURED = false`

`DUAL_MODEL_E2E_TEST = PARTIAL` (14 of 14 flows pass on CPU Triton 26.01; DINOv2 inference gated off; no GPU)

`UNIT_TESTS = PASS (435 tests, OK, skipped=3)`

`FRONTEND_CHECKS = PASS (lint 0 warnings, tsc, next build)`

`FEATURE_BRANCH_UPDATED = true`

`PR_MERGED = false`

`DINOv3_PRODUCTION_ALLOWED = false`

`MODEL_CHANGED = false`

`CALIBRATION_CHANGED = false`

`FINAL_TEST_TOUCHED = false`

`PRODUCTION_PROMOTION_ALLOWED = false`

## 1. Diagnostics

Raw output: `docs/engineering/runtime_diagnostics_2026-10-09_phase51.txt`.

| Check | Result |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS (noble), kernel 7.0.0-38-generic |
| `nvidia-smi` (host) | NVIDIA GeForce RTX 5080, driver 580.178.04, compute capability 12.0 |
| Docker | 29.4.3; runtimes `io.containerd.runc.v2`, `runc`; default `runc`; CDI spec dirs `/etc/cdi`, `/var/run/cdi` |
| `command -v nvidia-ctk` | not found (exit 1) — **absent package**, not a permission problem |
| `nvidia-ctk --version`, `nvidia-ctk cdi list` | command not found (exit 127) |
| `dpkg -l` NVIDIA container packages | none |
| `/etc/cdi`, `/var/run/cdi` | do not exist |
| `/etc/docker/daemon.json` | does not exist |
| `sudo -n true` | `sudo: a password is required` — **insufficient privilege** to install |
| Disk | 23 GB free on `/` |
| Running containers | 11 `fonoster-*` containers. A Docker restart would interrupt them |

## 2. Toolkit install: not performed

- The toolkit is genuinely missing, and installing it needs root.
- The agent did not request or store a password.
- The agent did not use docker-group access to modify the host through a privileged container, which would bypass the authentication requirement.
- The driver is unchanged.

The administrator commands are in `docs/engineering/gpu_runtime_admin_steps.md`, rewritten for this state from NVIDIA's install and CDI guides. The steps are:
1. Install toolkit 1.20.1-1.
2. Confirm `nvidia-ctk cdi list` shows `nvidia.com/gpu=all`.
3. Run `nvidia-ctk runtime configure --runtime=docker`, then restart Docker. This needs agreement from the `fonoster` owner.
4. Verify inside the pinned image.

## 3. GPU access inside Docker: hard stop

```text
$ docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.01-py3@sha256:c9f2ede5…146b nvidia-smi
docker: Error response from daemon: failed to discover GPU vendor from CDI: no known GPU vendor found
[exit=125]

$ docker run --rm --device nvidia.com/gpu=all nvcr.io/nvidia/tritonserver:26.01-py3@sha256:c9f2ede5…146b nvidia-smi
docker: Error response from daemon: CDI device injection failed: unresolvable CDI devices nvidia.com/gpu=all
[exit=125]
```

Concrete failure: Docker 29 resolves `--gpus` through CDI. There is no NVIDIA CDI specification and no `nvidia` runtime, so Docker finds no GPU vendor. Both are created by the NVIDIA Container Toolkit, which is absent.

The image digest was verified: `docker image inspect` reports `Id` and `RepoDigests` `sha256:c9f2ede50ccc4a3ce66e22e26ed1b5adc0c68516b58edfaca738693719c2146b`, created 2026-01-27. The CUDA test operation was not run because no container can see the GPU.

## 4. Triton on the pinned image, CPU only (not GPU evidence)

Run with the model folder mounted read-only, explicit model control, only `dinov2_vitb14_live` and `dinov3_authenticity_candidate` requested, and the tracked `KIND_CPU` configs installed. The container logged `WARNING: The NVIDIA Driver was not detected. GPU functionality will not be available.`

| Check | Result |
| --- | --- |
| Server live / ready | 200 / 200 |
| `dinov2_vitb14_live` v1 | `READY` (Triton 23.10 had failed: IR 10 > 9) |
| `dinov3_authenticity_candidate` v1 | `READY` |
| `dinov2_classifier` (old ViT-g/14) | not loaded (ready 400; absent from the loaded set) |
| Duplicate `dinov3_authenticity_classifier` | not loaded |

Checkpoint identity verified before use:
- `best_model.pt` hashes to `fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66`, equal to `infra/triton/dinov2_vitb14_live/identity.json`.
- `epoch_018.pt` hashes to `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`.

## 5. Parity

**DINOv2 GPU parity: BLOCKED.** **DINOv3 GPU parity: BLOCKED.** No GPU instance exists to run `--require-gpu` against.

The CPU DINOv3 result from Phase 50 was not repeated or relabelled.

DINOv2 CPU Triton parity was run once on 26.01, under the v1 protocol (`8b7f7bbf…854b`), which pins the `KIND_CPU` config. Output: `dinov2_live_triton_cpu_parity.json`.

| Batch size | Samples | Max abs logit error | Mean abs logit error | Max abs probability error | Verdict mismatches vs FP32 / live FP16 |
| --- | --- | --- | --- | --- | --- |
| 1 | 24 | 1.621e-05 | 5.235e-06 | 4.141e-07 | 0 / 0 |
| 2 | 24 | 1.621e-05 | 5.235e-06 | 4.141e-07 | 0 / 0 |
| 4 | 24 | 1.621e-05 | 5.235e-06 | 4.141e-07 | 0 / 0 |
| 8 | 24 | 1.621e-05 | 5.235e-06 | 4.141e-07 | 0 / 0 |

- Tolerances (`1e-4` logit, `1e-5` probability) are unchanged.
- The result is `routing_allowed_by_this_result: false`. The pre-registered routing rule allows the selector to use DINOv2 only after GPU parity, so `validated_instance_kinds` for DINOv2 stays empty.
- CPU and GPU Triton results now go to separate files (`dinov2_live_triton_cpu_parity.json`, `dinov2_live_triton_gpu_parity.json`).

## 6. Performance and stability

Not measured. The plan measures only after GPU parity passes. No CPU numbers were substituted.

## 7. Integration tests (PARTIAL)

`scripts/e2e_dual_model.py` ran against: Postgres and Redis, a research-mode API, a production-mode API, CPU Triton 26.01, and `next start`. Output: `integration_test_results_phase51_cpu.json`. All 14 flows pass. The E2E accounts were deleted afterwards (53 listings and 8 users before and after).

| Requirement | Result |
| --- | --- |
| DINOv2 routes to `dinov2_vitb14_live` | Gate held. Readiness `PARITY_NOT_VALIDATED_FOR_SERVED_INSTANCE`, HTTP 503, `decision: null`. `dinov2_vitb14_live` inference count 96 → 96 and DINOv3 0 → 0. **DINOv2 inference itself: BLOCKED** |
| Old ViT-g/14 never used | `dinov2_classifier` not loaded and not called |
| DINOv3 routes to `dinov3_authenticity_candidate` | HTTP 200; inference count 0 → 1; `model = DINOV3_RESEARCH_PROTOTYPE`, version 1, frozen checkpoint SHA |
| Unavailable model | Unload → HTTP 503 `MODEL_ERROR`, `decision: null`; reload → same decision |
| No silent fallback | Shown by the inference counts above |
| DINOv3 blocked in production mode | HTTP 403 `POLICY_ERROR`, `decision: null`, on both the research route and the live route |
| Unsupported brand, missing brand, invalid image, Triton name as selector | 422 / 422 / 400 / 422, all `decision: null` |
| No automatic publication | Research calls created 0 listings; a legacy check stored `pending` |
| Historical labels | 19 × `Legacy screening — not verified`, 20 × `Demo listing — not verified` |
| Research allowlist | Buyer 403, anonymous 401, `admin` registration rejected |

Browser check (production build, entitled user):
- DINOv2 — Legacy is disabled with "unavailable (Triton parity not validated for this instance)" and "same model as the live listing check".
- DINOv3 is selected and shows `EXPERIMENTAL — NOT APPROVED FOR PRODUCTION`.

## 8. Regression

- `.venv/bin/python -m unittest discover -s tests -v`: 435 tests, OK. The 3 skips are environmental: Triton not on port 8001, ONNX CUDA provider libraries missing, TensorRT/Triton unavailable.
- Frontend: `npm run lint` passed with no warnings or errors; `npx tsc --noEmit` passed; `npm run build` compiled.

## 9. Git

- Branch `feature/dual-model-triton`. Started this phase at `f0e5cf1` (matched `origin`). Evidence commit `a33130f`, plus the commit that adds this report.
- Staged by explicit path only. No `.env`, credentials, datasets, weights, or `models/` files are committed.
- Frozen checkpoints unchanged. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False`.
- `main` = `e0c0c6d27afa5275909ff3c9a3fbe0713a28f330`, unchanged. PR #1 stays open and draft. Not merged. No deployment.

The independent data pilot remains `NO_GO`. Successful serving would not be evidence that authenticity generalizes across unseen brands.

## Remaining blocker and next step

The only blocker is the NVIDIA Container Toolkit install. An administrator needs to run steps 1–4 of `docs/engineering/gpu_runtime_admin_steps.md`, after agreeing a Docker restart window with the owner of the running `fonoster` containers.

Once `nvidia-smi` inside the pinned image shows the RTX 5080, the step 5 runbook completes the work:
- GPU configs installed, Triton started on the GPU
- both frozen GPU parity protocols run
- GPU metrics, latency, memory, and stability measured
- E2E re-run on the GPU stack
- a reviewed commit adding `KIND_GPU` for each model that passes
