# Dual-model Triton architecture

DINOv2 and DINOv3 stay separate. The browser sends a logical identifier. The server maps that identifier to one Triton model. There is no fallback.

## Routes

- `POST /verify/authenticate` stays on `dinov2_classifier`. A request that names `dinov3_experimental` is rejected with `POLICY_ERROR` and no inference.
- `POST /research/verify` accepts `dinov2_legacy` or `dinov3_experimental`. Omitting the field keeps the existing DINOv3 research default.
- `GET /research/models` reports readiness for each model and does not return a decision.
- Research mode is required for `/research/verify`. Production, missing, and unknown deployment modes return `POLICY_ERROR`.
- Neither research response creates or publishes a listing. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED` remains false.

## Adapters

| Logical id | Triton model | Version | Input | Preprocess | Output |
| --- | --- | --- | --- | --- | --- |
| `dinov2_legacy` | `dinov2_classifier` | 1 | FP32 `[3, 518, 518]` | legacy square resize and ImageNet normalization | raw logit |
| `dinov3_experimental` | `dinov3_authenticity_candidate` | 1 | FP32 `[3, 512, 512]` | `resize_pad_square_eval_v1` | raw logit |

DINOv3 temperature `0.24038200410185356` and decision threshold `0.50` stay outside the graph. They are not applied to DINOv2.

## Repository layout

- Existing `models/dinov2_classifier` is unchanged.
- Existing `models/dinov3_authenticity_classifier` export is unchanged.
- `dinov3_authenticity_candidate` is a separate name. Its local version-1 ONNX entry is a symlink to that export. The tracked config is `infra/triton/dinov3_authenticity_candidate/config.pbtxt`.
- The candidate instance group is `KIND_CPU`. GPU was not enabled because Docker cannot pass the RTX 5080 into the Triton image.

## Current serving state

The host has an RTX 5080 and CUDA PyTorch. `tritonserver` is not installed on the host. The local image `nvcr.io/nvidia/tritonserver:23.10-py3` cannot start with `--gpus all` because Docker reports `failed to discover GPU vendor from CDI: no known GPU vendor found`. No NVIDIA container runtime is registered. This phase did not install drivers or a container toolkit.

Until a compatible Triton process can load each model, both readiness flags stay false and the research route returns `MODEL_ERROR` with `decision: null`.
