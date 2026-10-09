# Dual-model Triton architecture

DINOv2 and DINOv3 stay separate. The browser sends a logical identifier. The server maps that identifier to one Triton model and version. There is no fallback.

## Routes

- `POST /verify/authenticate` is the live listing check. It runs local PyTorch `vit_base_patch14_dinov2.lvd142m` at 504 (`INFERENCE_BACKEND=torch`, float16 autocast on CUDA). A request that names `dinov3_experimental` is rejected with `POLICY_ERROR` before inference.
- `POST /research/verify` accepts `logical_model` = `dinov2_legacy` or `dinov3_experimental`. Omitting it keeps the existing DINOv3 default.
- `GET /research/models` reports Triton server liveness and, per model, readiness, served instance kind, validated instance kinds, and the reason a model is unavailable. It returns no decision.
- Both research endpoints require an entitled account (`RESEARCH_USER_EMAILS`). They also require `HYPEVAULT_DEPLOYMENT_MODE` set to `research` or `shadow`. Production, missing, and unknown modes return `POLICY_ERROR`.
- No research response creates or publishes a listing. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED` remains false.

## Adapters

| Logical id | Triton model | Version | Input | Preprocess | Output | Validated instance kinds |
| --- | --- | --- | --- | --- | --- | --- |
| `dinov2_legacy` | `dinov2_vitb14_live` | 1 | FP32 `[3, 504, 504]` | live route's `preprocess_chw` at 504 | raw logit | none yet |
| `dinov3_experimental` | `dinov3_authenticity_candidate` | 1 | FP32 `[3, 512, 512]` | `resize_pad_square_eval_v1` | raw logit | `KIND_CPU` |

DINOv2 uses the live route's verdict: sigmoid at 0.50, then the 0.88 minimum-authentic-confidence floor. DINOv3 uses `shadow_v1` with temperature `0.24038200410185356` and threshold `0.50` outside the graph. Neither model receives the other's calibration.

## Validation gate

A model is selectable only when Triton reports it ready and every instance kind it is served on has passed Triton parity. Otherwise the selector disables it and shows the reason, and `/research/verify` returns `MODEL_ERROR` with `decision: null`.

## Repository layout

- `models/dinov2_vitb14_live/1` holds byte-identical copies of the live export. Its config is tracked at `infra/triton/dinov2_vitb14_live/config.pbtxt` (`KIND_CPU`) and `config.gpu.pbtxt` (`KIND_GPU`), and its identity at `identity.json`.
- `models/dinov2_classifier` (ViT-g/14 at 518) is unchanged and no longer routed.
- `models/dinov3_authenticity_classifier` is the single authoritative DINOv3 export. `dinov3_authenticity_candidate` version 1 symlinks to it. Explicit model control keeps the duplicate folder unloaded.

## Serving state on 2026-10-09

| Check | Result |
| --- | --- |
| DINOv2 export identity (host ONNX Runtime) | PASS: strict checkpoint load; max logit error `1.62e-5` vs FP32; 0 verdict mismatches vs FP32 and live FP16 |
| DINOv2 on Triton 23.10 | `UNAVAILABLE`: ONNX IR 10 exceeds the IR 9 maximum of ONNX Runtime 1.16 |
| DINOv3 on CPU Triton 23.10 | `READY`; parity PASS on 24 non-test inputs at batch sizes 1, 2, 4, 8 |
| GPU containers | Blocked: no NVIDIA Container Toolkit, no CDI spec, no passwordless sudo |
| Pinned replacement image | `nvcr.io/nvidia/tritonserver:26.01-py3@sha256:c9f2ede5…146b`. Not pulled, waiting on the GPU runtime |
