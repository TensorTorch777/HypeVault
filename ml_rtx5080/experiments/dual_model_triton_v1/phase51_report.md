# Phase 51 report — NVIDIA Docker GPU access and Triton validation

GPU access inside Docker works. Both selector models serve on the RTX 5080 through the pinned Triton 26.01 image, and both pass their frozen GPU parity protocols.

The first GPU attempt failed parity for both models. ONNX Runtime's CUDA provider uses TF32 math by default, which breaks the exact-FP32 contract. That failure is recorded. The fix is an exact-FP32 serving config (`use_tf32=0`) with tolerances unchanged, re-run under protocols committed before measurement.

The research selector serves a model only on the validated GPU precision.

## Flags

`NVIDIA_TOOLKIT_INSTALLED = true (1.20.1, commit dffc40b4)`

`DOCKER_GPU_ACCESS = PASS`

`RTX5080_VISIBLE_IN_CONTAINER = true (GPU 0: NVIDIA GeForce RTX 5080, UUID GPU-b415f453-8f9a-ea64-7780-65d3d8086109, driver 580.178.04)`
  
`TRITON_VERSION = 2.65.0 (NGC 26.01, CUDA 13.1.1 in minor-version-compatibility mode on the CUDA 13.0 driver; image sha256:c9f2ede50ccc4a3ce66e22e26ed1b5adc0c68516b58edfaca738693719c2146b)`

`DINOV2_TRITON_READY = true (dinov2_vitb14_live v1, KIND_GPU, use_tf32=0)`

`DINOV3_TRITON_READY = true (dinov3_authenticity_candidate v1, KIND_GPU, use_tf32=0)`

`DINOV2_GPU_PARITY = PASS (attempt 2 and dual-loaded re-run; attempt 1 FAIL under default TF32)`

`DINOV3_GPU_PARITY = PASS (attempt 2 and dual-loaded re-run; attempt 1 FAIL under default TF32)`

`GPU_PERFORMANCE_MEASURED = true`

`DUAL_MODEL_E2E_TEST = PASS (14 of 14 flows on the GPU stack, repeated after digest cache)`

`UNIT_TESTS = PASS (444 tests, OK, skipped=3; 9 new checkpoint-identity tests)`

`FRONTEND_CHECKS = PASS (lint 0 warnings, tsc, next build)`

`FEATURE_BRANCH_UPDATED = true`

`PR_MERGED = true (merge commit f2acf4a211541fc509f67fc2a1157d4dbac83d16)`

`DINOv3_PRODUCTION_ALLOWED = false`

`MODEL_CHANGED = false`

`CALIBRATION_CHANGED = false`

`FINAL_TEST_TOUCHED = false`

`PRODUCTION_PROMOTION_ALLOWED = false`

## 1. Toolkit install and Docker GPU access

- **Install:** the owner ran `scripts/install_nvidia_container_toolkit.sh` with sudo in their own terminal, following NVIDIA's install and CDI guides. The script refuses to continue if apt would change a driver package, and checks the driver version afterwards. Log: `/var/log/hypevault-nvidia-toolkit-2026-10-09-170611.log`.
- **Verified afterwards:**
  - `nvidia-ctk` 1.20.1 is installed.
  - `nvidia-ctk cdi list` shows `nvidia.com/gpu=0`, the GPU UUID, and `nvidia.com/gpu=all`.
  - `/etc/docker/daemon.json` registers the `nvidia` runtime; Docker runtimes are `io.containerd.runc.v2 nvidia runc`.
  - The host driver is unchanged at 580.178.04.
- **Inside the pinned image:**
  - `nvidia-smi -L` lists the RTX 5080: 16303 MiB, compute capability 12.0.
  - A CUDA driver call inside the container (`cuInit`) returned 0 and found 1 device; driver CUDA version 13000.
- **Image identity:** `docker image inspect` reports Id and RepoDigest `sha256:c9f2ede5…146b`.

## 2. Triton on the GPU

Triton ran with explicit model control and only `dinov2_vitb14_live` and `dinov3_authenticity_candidate` loaded. The model folder was mounted read-only.

| Check | Result |
| --- | --- |
| Server live / ready | 200 / 200 |
| `dinov2_vitb14_live` v1 | READY, instance `KIND_GPU` on GPU 0 |
| `dinov3_authenticity_candidate` v1 | READY, instance `KIND_GPU` on GPU 0 |
| `dinov2_classifier` (old ViT-g/14) | Not loaded, never called (E2E flow 2b) |
| Duplicate `dinov3_authenticity_classifier` export | Not loaded |
| GPU execution | Triton logs `ModelInstanceInitialize … (GPU device 0)` and memcpy nodes for `CUDAExecutionProvider`; `nv_gpu_utilization` peaked at 0.83–0.96 during the parity runs |

Checkpoint identity was re-checked before use:
- `best_model.pt` = `fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66`, equal to `identity.json`.
- `epoch_018.pt` = `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`.

## 3. GPU parity

Tolerances were never changed: max abs logit error `1e-4`, max abs probability error `1e-5`, 0 decision mismatches. All runs used the same 24 non-test inputs at batch sizes 1, 2, 4 and 8, against FP32 CPU PyTorch references.

### Attempt 1 — FAIL (reviewed GPU configs, ONNX Runtime default TF32)

| Model | Protocol | Max abs logit error | Max abs probability error | Decision mismatches |
| --- | --- | --- | --- | --- |
| DINOv2 | `dinov2_live_triton_gpu_protocol_v1.json` (`c4ec2851…`) | 1.379e-03 | 2.916e-05 | 0 (and 0 vs live FP16) |
| DINOv3 | `parity_protocol_v1.json` (`858abd0c…`) | 4.773e-02 | 1.216e-03 | 0 |

Results: `dinov2_live_triton_gpu_parity_attempt1.json` and `dinov3_triton_gpu_parity_attempt1.json`.

### Investigation

- **Hypothesis source:** ONNX Runtime's CUDA EP documentation states TF32 is enabled by default, and `use_tf32=0` disables it from 1.18. The image ships 1.24.1.
- **Test design:** I ran `ml_rtx5080/triton_precision_diagnostic.py` on 8 seeded synthetic images, not the protocol inputs, so those stayed unseen for any new config. It used a scratch Triton repository of hard links in `/tmp`.

| Model | TF32 default, max abs logit error | `use_tf32=0`, max abs logit error |
| --- | --- | --- |
| DINOv2 | 3.07e-04 | 5.72e-06 |
| DINOv3 | 4.94e-04 | 2.62e-06 |

Result: `gpu_precision_diagnostic.json`.

### Attempt 2 — PASS (exact-FP32 GPU configs, `use_tf32=0`)

New reviewed configs: `infra/triton/*/config.gpu_fp32.pbtxt`, the GPU config plus `use_tf32=0`. The v2 protocols were committed in `2d790b2` before this run:
- DINOv2 `dinov2_live_triton_gpu_protocol_v2.json`, SHA-256 `0c36cb55…de52`
- DINOv3 `dinov3_triton_gpu_protocol_v2.json`, SHA-256 `4db66874…bd1d`

Inputs, tolerances, and pass rules are identical to v1; only the pinned config differs. Both harnesses refuse to run unless the config Triton loaded is byte-identical to the pinned one.

| Model | Batch | Samples | Max abs logit error | Mean abs logit error | Max abs probability error | Mismatches |
| --- | --- | --- | --- | --- | --- | --- |
| DINOv2 | 1 | 24 | 2.360e-05 | 5.494e-06 | 4.460e-07 | 0 vs FP32 / 0 vs live FP16 |
| DINOv2 | 2 | 24 | 2.337e-05 | 5.563e-06 | 4.229e-07 | 0 / 0 |
| DINOv2 | 4 | 24 | 2.360e-05 | 5.662e-06 | 4.272e-07 | 0 / 0 |
| DINOv2 | 8 | 24 | 2.360e-05 | 5.345e-06 | 4.272e-07 | 0 / 0 |
| DINOv3 | 1 | 24 | 5.245e-06 | 8.345e-07 | 5.330e-13 | 0 |
| DINOv3 | 2 | 24 | 4.256e-05 | 2.886e-06 | 9.813e-07 | 0 |
| DINOv3 | 4 | 24 | 3.648e-05 | 3.268e-06 | 8.409e-07 | 0 |
| DINOv3 | 8 | 24 | 1.788e-05 | 2.116e-06 | 4.122e-07 | 0 |

Results: `dinov2_live_triton_gpu_parity_attempt2.json` and `dinov3_triton_gpu_parity_attempt2.json`. Triton model version 1 for both. `nv_gpu_utilization` peaked at 0.88 during the DINOv2 run.

## 4. Performance and stability (GPU, measured after parity)

Method (`ml_rtx5080/triton_gpu_benchmark.py`; results in `gpu_performance_results.json`):
- Each model was measured alone, with the other unloaded.
- One gRPC client ran on the host.
- The input was the first non-test image of the parity protocol, tiled for larger batches.
- Client round trip includes serialization and transport; "Triton compute" is Triton's own `compute_infer` time.

| Metric | DINOv2 (`dinov2_vitb14_live`) | DINOv3 (`dinov3_authenticity_candidate`) |
| --- | --- | --- |
| Warm batch-1 client latency p50 / p99 | 20.11 / 21.73 ms | 16.73 / 17.31 ms |
| Triton compute per request (batch 1) | 17.67 ms | 14.38 ms |
| Throughput batch 1 / 2 / 4 / 8 | 49.3 / 50.4 / 42.2 / 40.1 img/s | 59.2 / 64.4 / 57.5 / 54.9 img/s |
| Request time batch 8 | 199.4 ms | 145.6 ms |
| GPU memory (tritonserver): no model / warm / peak after batch 8 | 392 / 1546 / 9738 MiB | 392 / 1546 / 3594 MiB |
| Sequential stability (200 requests) | 0 errors, max deviation 0 | 0 errors, max deviation 0 |
| Concurrent stability (8 threads × 25) | 200/200 completed, 0 errors, 54.8 req/s, p50 145.6 ms, deviation 0 | 200/200, 0 errors, 68.3 req/s, p50 116.7 ms, deviation 0 |
| Load / reload | 0.11 s / 0.24 s; not-ready while unloaded; identical output after reload | 0.39 s / 0.31 s; same |
| API end-to-end `/research/verify` p50 / p90 (30 requests) | 45.9 / 49.0 ms | 495.3 / 499.4 ms |

Observations:
- **Batching:** exact FP32 gives little throughput gain from batching on this GPU.
- **DINOv2 peak memory:** grows to 9.7 GB at batch 8 because ONNX Runtime's memory arena expands.
- **DINOv3 API latency:** the 495 ms is far above its 14 ms Triton compute. The research route re-hashes the 1 GB frozen checkpoint (`frozen_checkpoint_digest`) on every DINOv3 request. This is a follow-up item; it was not changed here.

## 5. Application integration (GPU stack)

`scripts/e2e_dual_model.py` ran against Postgres and Redis, a research-mode API, a production-mode API, GPU Triton, and `next start`. Output: `integration_test_results_phase51_gpu.json`. All 14 flows pass.

| Requirement | Evidence |
| --- | --- |
| DINOv2 routes to `dinov2_vitb14_live` | HTTP 200 `LEGACY_DINOV2`; inference count 31 → 32; DINOv3 1 → 1 |
| DINOv3 routes to `dinov3_authenticity_candidate` | HTTP 200 `DINOV3_RESEARCH_PROTOTYPE`, version 1, frozen checkpoint SHA; count 1 → 2 |
| Model that ran is identified | `model`, `model_version`, and `checkpoint_sha` in each response |
| Unavailable model | DINOv3 unloaded → HTTP 503 `MODEL_ERROR`, `decision: null`; reload → same decision |
| No silent fallback | Inference counts above; old ViT-g/14 never called |
| DINOv3 blocked in production | HTTP 403 `POLICY_ERROR`, `decision: null`, on both the research and live routes |
| Unsupported brand, missing brand, invalid image, Triton name as selector | 422 / 422 / 400 / 422, all `decision: null` |
| No automatic publication | Research calls created 0 listings; the legacy check stored `pending` |
| Historical labels | 19 × `Legacy screening — not verified`, 20 × `Demo listing — not verified` |
| Research allowlist | Buyer 403, anonymous 401, `admin` registration rejected |

Browser check (production build, entitled user):
- Both models show "ready".
- DINOv2 reads "same model as the live listing check".
- Selecting DINOv3 shows `EXPERIMENTAL — NOT APPROVED FOR PRODUCTION`.

The E2E accounts were deleted afterwards (53 listings and 8 users before and after).

Gate change (`fafa0fa`):
- A model is selectable on `KIND_GPU` only if Triton serves it with `use_tf32=0`.
- A TF32-default GPU instance is reported `SERVING_PRECISION_NOT_VALIDATED`. Unit tests cover this case.
- DINOv2 is not validated on CPU, so it is never routed there.

## 6. Regression

- `.venv/bin/python -m unittest discover -s tests -v`: 435 tests, OK. The 3 environmental skips are old tests that look for Triton on port 8001, the ONNX CUDA provider libraries on the host, and TensorRT.
- Frontend: `npm run lint` passed with no warnings or errors; `npx tsc --noEmit` passed; `npm run build` compiled.

## 7. Side effect of the Docker restart (outside HypeVault)

After the restart, 4 of the 11 `fonoster` containers did not come back: `apiserver`, `routr`, `autopilot`, `envoy`. Each fails with `error mounting "/home/tensortorch26/Desktop/Voice_hosp/fonoster/config/…": not a directory`.

The compose project lives at `/home/tensortorch26/Desktop/Voice_hosp/fonoster`, which did not exist when Docker restarted. Docker created empty root-owned directories at 17:06:39 in place of the missing `integrations.json`, `envoy.yaml`, and `keys/public.pem`. Those containers had been running on mounts whose source files were already gone, and the restart exposed that.

The agent did not modify that project. To fix it, restore the `fonoster` project files to that path (or start the stack from its current location), remove the empty placeholder directories with sudo, and run `docker compose up -d` in the project.

## 8. Git

- Branch `feature/dual-model-triton`. Phase commits: `a33130f`, `ec01d17` (CPU-blocked state), `2d790b2` (attempt-1 failures, diagnostic, v2 protocols), `93e55c6` (GPU parity PASS), `fafa0fa` (gate and E2E), plus the commit that adds this report.
- Staged by explicit path. No `.env`, credentials, datasets, weights, or `models/` files are committed. Frozen checkpoints are unchanged. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False`.
- `main` = `e0c0c6d27afa5275909ff3c9a3fbe0713a28f330`, unchanged. PR #1 stays open and draft. Not merged. No deployment.
- The local `models/*/config.pbtxt` now hold the exact-FP32 GPU configs (gitignored). The CPU configs remain tracked in `infra/triton/*/config.pbtxt`.

The independent data pilot remains `NO_GO`. Successful GPU serving and parity are not evidence that authenticity generalizes across unseen brands.

Fonoster is out of scope for this repository and was not restored, modified, or deleted.

## 9. Checkpoint digest cache and dual-model concurrency

Follow-up on `feature/dual-model-triton`. The frozen checkpoint, temperature `0.24038200410185356`, threshold `0.50`, and final-test artifacts were not changed.

### 9.1 Hash verification moved to startup

`backend/inference/checkpoint_identity.py` hashes `epoch_018.pt` once during FastAPI lifespan (and again only on an explicit reload). The digest is cached only if it equals `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`. A mismatch or hashing error leaves the cache empty; `/research/verify` then returns `decision = null` and does not call Triton. DINOv2 verification is unchanged.

| Measurement | Before | After |
| --- | --- | --- |
| SHA-256 of the 1 046 865 163-byte checkpoint | 444.3 ms mean (3 runs; `checkpoint_hash_before.json`) | 444.1 ms once at process start (`checkpoint_hash_after.json`) |
| Cached digest lookup | n/a (hashed every request) | 0.000086 ms mean over 100 lookups; `hash_calls = 1` |
| `/research/verify` DINOv3 API p50 / p90 (30 requests) | 495.3 / 499.4 ms (`gpu_performance_results.json`) | 44.3 / 49.6 ms sequential with both models loaded (`dual_model_concurrency.json`); E2E repeat 44.3 / 50.6 ms (`integration_test_results_digest_cache.json`) |
| Triton `compute_infer` during those 30 DINOv3 API requests | 14.38 ms (single-model bench) | 14.42 ms |
| DINOv2 API p50 / p90 (30 requests, both loaded) | 45.9 / 49.0 ms (DINOv2 alone) | 50.1 / 56.0 ms (first-request warmup 671 ms); E2E 45.7 / 48.5 ms |

The cached DINOv3 API p50 (44 ms) is still above Triton's 14 ms compute. The remaining time is HTTP, image decode, preprocessing, gRPC, and policy. After cache, `dinov3_checkpoint.hash_calls` stayed 1 through 30 sequential DINOv3 API calls plus 200 mixed concurrent DINOv3 calls.

Tests in `tests/test_checkpoint_identity.py`: correct identity is cached; repeated requests do not re-hash; a mismatched file fails closed and is not cached; reload re-hashes; hashing errors return `MODEL_ERROR` with `decision = null`.

### 9.2 Dual-loaded mixed concurrency

Both models were loaded together on the same Triton 26.01 process, GPU 0, `KIND_GPU`, `use_tf32=0`. Evidence: `dual_model_concurrency.json`.

| Memory | MiB |
| --- | --- |
| Idle Triton (no models) | 324 |
| DINOv2 only | 920 |
| Dual loaded, idle | 1 506 |
| Peak Triton during mixed API load | 2 706 |
| Host GPU used after mixed load | 3 222 |

16 303 MiB GPU. Dual-loaded mixed traffic fit. No OOM, no precision change.

Mixed concurrent `/research/verify` (8 workers, 200 requests per model, explicit `logical_model`):

| Model | Completed | Failed | p50 / p90 ms | Triton inference-count delta | Mean compute_infer ms |
| --- | --- | --- | --- | --- | --- |
| `dinov2_legacy` → `dinov2_vitb14_live` | 200 | 0 | 126.0 / 135.0 | 200 | 17.64 |
| `dinov3_experimental` → `dinov3_authenticity_candidate` | 200 | 0 | 116.7 / 186.6 | 200 | 14.53 |

Combined throughput 52.7 req/s. Decisions consistent (`AUTHENTIC` on the protocol image). DINOv3 responses carried the frozen checkpoint SHA. Unload of one model left the other READY; reload restored both. Old `dinov2_classifier` was not called.

GPU parity re-run with **both** models resident, same frozen tolerances (`1e-4` logit, `1e-5` probability):

- DINOv2: `dinov2_live_triton_gpu_parity_dual_loaded.json` PASS. Max abs logit error 2.360e-05, max abs probability error 4.460e-07, 0 verdict mismatches.
- DINOv3: `dinov3_triton_gpu_parity_dual_loaded.json` PASS. Max abs logit error 4.256e-05 (batch 2), max abs probability error 9.813e-07, 0 decision mismatches.

### 9.3 Regression

- `.venv/bin/python -m unittest discover -s tests -v`: first pass while Triton still held a 12.6 GiB batch-8 arena failed `test_tiny_training_smoke` with CUDA OOM. After unloading both models (Triton 392 MiB), **444 tests, OK, skipped=3**.
- Frontend: `npm run lint` 0 warnings; `npx tsc --noEmit`; `npm run build` compiled.
- E2E `scripts/e2e_dual_model.py` against research API :8000, production API :8010, GPU Triton, `next start`: **14/14 PASS** (`integration_test_results_digest_cache.json`). Live `/verify/authenticate` used `INFERENCE_BACKEND=torch` (the research selector still uses Triton). Historical labels 19 × `Legacy screening — not verified`, 20 × `Demo listing — not verified`. Listings 53 before and after. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = false`.

### 9.4 Git

Intended files only. No `.env`, weights, datasets, or unrelated `ml_rtx5080/train.py` / experiment trees.

- Merge commit: `f2acf4a211541fc509f67fc2a1157d4dbac83d16` (PR [#1](https://github.com/TensorTorch777/HypeVault/pull/1), merged 2026-10-09, no branch protection / required reviews on `main`).
- `main` verified at that SHA: checkpoint identity module present, `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = false`, 41 routing/identity/publication tests OK. GitHub CI on `6e05e66` (backend + frontend) SUCCESS; GPU job SKIPPED (manual).
- Research branch `research/dinov3-cross-brand-v2` created from that `main`. Plan: `docs/research/dinov3_cross_brand_v2_plan.md`. No training started.

## Flag block

`CHECKPOINT_DIGEST_CACHED_AT_STARTUP = true`

`HASH_MISMATCH_FAILS_CLOSED = true`

`BOTH_MODELS_READY_SIMULTANEOUSLY = true`

`DUAL_MODEL_GPU_CONCURRENCY = PASS`

`DINOV2_GPU_PARITY = PASS`

`DINOV3_GPU_PARITY = PASS`

`DINOv3_API_LATENCY_IMPROVED = true (p50 495.3 ms → 44.3 ms; not equal to 14.4 ms Triton compute)`

`AUTOMATIC_PUBLICATION_BLOCKED = true`

`DINOv3_PRODUCTION_ALLOWED = false`

`PR_MERGED = true (f2acf4a211541fc509f67fc2a1157d4dbac83d16)`

`MAIN_VERIFIED = true`

`RESEARCH_BRANCH_CREATED = true (research/dinov3-cross-brand-v2)`

`MODEL_CHANGED = false`

`CALIBRATION_CHANGED = false`

`FINAL_TEST_TOUCHED = false`

`PRODUCTION_PROMOTION_ALLOWED = false`
