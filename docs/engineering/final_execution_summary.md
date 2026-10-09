# Final execution summary

Your three decisions are applied.

- **DINOv2:** the selector's DINOv2 option now targets the live ViT-B/14 at 504, registered as the separate Triton model `dinov2_vitb14_live`. Its artifact identity and export parity are verified. `dinov2_classifier` is untouched and no longer routed.
- **GPU:** both models are waiting on the administrator's GPU container setup, which this session cannot do without root.
- **PR #1:** stays open and unmerged. Nothing was trained, recalibrated, re-thresholded, or deployed, and the locked final test was not opened.

## Status contract

| State | Value | Basis |
| --- | --- | --- |
| `REPOSITORY_RELEASE_READY` | `false` | CPU CI and clean-checkout tests pass. The merge waits on Triton GPU validation of both models, per your decision. |
| `TRITON_GPU_RUNTIME_READY` | `false` | Re-checked: `nvidia-ctk` absent, no CDI spec, `docker run --gpus all` fails with the CDI error. |
| `DINOV2_TRITON_READY` | `false` | `dinov2_vitb14_live` export parity passes, but Triton 23.10 cannot load ONNX IR 10, and the GPU path is blocked. The selector shows it unavailable. |
| `DINOV3_TRITON_READY` | `false` | Ready and parity-passing on CPU Triton only. GPU parity is pending. |
| `DUAL_MODEL_E2E_TESTED` | `false` | DINOv2 has never served through Triton. The earlier 13-flow E2E run predates the remap and must be repeated on the GPU stack. |
| `DATASET_READY_FOR_TRAINING` | `false` | Data `NO_GO`. |
| `CROSS_BRAND_VALIDATION_COMPLETE` | `false` | Gated on verified data. |
| `PRODUCTION_PROMOTION_ALLOWED` | `false` | `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False`. `DINOv3_PRODUCTION_ALLOWED = false`. |

## Evidence for Option B

- **Identity:** `ml_rtx5080/checkpoints/best_model.pt` (SHA-256 `fe1daa0b…fa66`) loads strictly into the live `DINOv2Classifier` (0 missing, 0 unexpected keys). The Triton copies of `model.onnx` (`f367c2d4…7c17c`) and its external data (`b0a57707…172c`) are byte-identical to the 2026-05-15 export. Recorded in `infra/triton/dinov2_vitb14_live/identity.json`.
- **Export parity:** host ONNX Runtime on the exact Triton bytes, 24 non-test inputs, batch sizes 1, 2, 4, 8. Pre-registered in `6e5856f`.
  - max abs logit error vs FP32 `1.62e-5` (tolerance `1e-4`)
  - max abs probability error `4.14e-7` (tolerance `1e-5`)
  - 0 verdict mismatches vs FP32 and vs the live FP16 GPU path
- **Gate:**
  - A model is selectable only on a Triton instance kind that passed Triton parity. DINOv2 has none, so it is unavailable.
  - DINOv3 is validated for `KIND_CPU` only, so serving it on `KIND_GPU` makes it unavailable until GPU parity passes.
  - Against live Triton 23.10, a DINOv2 request returned `MODEL_UNAVAILABLE`, and the DINOv3 inference count stayed 0 → 0.

## Blocked

| Item | Exact dependency |
| --- | --- |
| GPU containers | An administrator runs steps 2–4 of `docs/engineering/gpu_runtime_admin_steps.md`. |
| DINOv2 and DINOv3 Triton GPU parity, latency, memory, E2E | After that, run the step 5 commands. Both protocols are frozen: DINOv2 GPU `c4ec2851…e710`, DINOv3 `858abd0c…e4cb`. |
| Selector on GPU | A reviewed commit adding `KIND_GPU` to each model's `validated_instance_kinds`, citing the passing results. |
| Merge | Both models pass Triton GPU validation, then your review of PR #1. |
| Data, research, promotion | Unchanged: needs a verified-data partner and permissions. |

## Branch and PR state

- `main` = `origin/main` = `e0c0c6d27afa5275909ff3c9a3fbe0713a28f330`, unchanged.
- `feature/dual-model-triton`: `ac17a87` → `4b71af0` → `58945be` → `93ee6dd` → `7c9501b` → `b639dc3` → `8cd2cab` → `f688ff7` → `af4c360` → `6e5856f` → `ad721c9`, plus the commit that adds this update.
- PR #1: open, draft. Not merged. No deployment.

## Next action required from you

Have an administrator run steps 2–4 of `docs/engineering/gpu_runtime_admin_steps.md`, then tell me. I will confirm `nvidia-smi` inside the pinned 26.01 container, run both GPU parity protocols, measure latency and memory, repeat the E2E flows on the GPU stack, and ask for your review only if both models pass.
