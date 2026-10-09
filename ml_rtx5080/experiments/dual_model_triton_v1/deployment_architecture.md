# Dual-model Triton architecture

DINOv2 and DINOv3 stay separate. The browser sends a logical identifier. The server maps that identifier to one Triton model and version. There is no fallback.

## Routes

- `POST /verify/authenticate` is the live listing check. It runs local PyTorch `vit_base_patch14_dinov2.lvd142m` at 504 (`INFERENCE_BACKEND=torch`), not Triton. A request that names `dinov3_experimental` is rejected with `POLICY_ERROR` before inference.
- `POST /research/verify` accepts `logical_model` = `dinov2_legacy` or `dinov3_experimental`. Omitting it keeps the existing DINOv3 default.
- `GET /research/models` reports Triton server liveness and readiness for each model. It returns no decision.
- Both research endpoints require an entitled account (`RESEARCH_USER_EMAILS`). They also require `HYPEVAULT_DEPLOYMENT_MODE` set to `research` or `shadow`. Production, missing, and unknown modes return `POLICY_ERROR`.
- No research response creates or publishes a listing. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED` remains false.

## Adapters

| Logical id | Triton model | Version | Input | Preprocess | Output |
| --- | --- | --- | --- | --- | --- |
| `dinov2_legacy` | `dinov2_classifier` | 1 | FP32 `[3, 518, 518]` | square bicubic resize to 518, ImageNet normalization | raw logit |
| `dinov3_experimental` | `dinov3_authenticity_candidate` | 1 | FP32 `[3, 512, 512]` | `resize_pad_square_eval_v1` | raw logit |

DINOv3 temperature `0.24038200410185356` and threshold `0.50` stay outside the graph. DINOv2 uses the legacy sigmoid verdict plus the live route's 0.88 minimum-authentic-confidence floor. No DINOv3 calibration is applied to DINOv2.

## DINOv2 identity

`dinov2_classifier` is a 40-block ViT-g/14 export at 518, dated 2026-04-27. The live route runs a different model: ViT-B/14 at 504, from 2026-05-15. The selector reports `same_model_as_live_route: false` for this reason. Which DINOv2 should back the research selector is an owner decision.

## Repository layout

- `models/dinov2_classifier` is unchanged. A byte-identical config is tracked at `infra/triton/dinov2_classifier/config.pbtxt`.
- `models/dinov3_authenticity_classifier` is the single authoritative DINOv3 export. Its `model.onnx` SHA-256 is `d1d0c9bc…3edb`, matching `identity.json`.
- `dinov3_authenticity_candidate` version 1 symlinks to that export. Its tracked config is `infra/triton/dinov3_authenticity_candidate/config.pbtxt`. Triton runs with explicit model control, so the duplicate export folder is never loaded.

## Serving state on 2026-10-09

| Check | Result |
| --- | --- |
| DINOv3 candidate on CPU Triton 23.10 | `READY`. Parity PASS on 24 non-test inputs at batch sizes 1, 2, 4, 8 |
| DINOv2 on Triton 23.10 | `UNAVAILABLE`: ONNX IR 10 exceeds the IR 9 maximum of ONNX Runtime 1.16 |
| GPU containers | Blocked. No NVIDIA Container Toolkit, no CDI spec, no passwordless sudo |
| Pinned replacement image | `nvcr.io/nvidia/tritonserver:26.01-py3@sha256:c9f2ede5…146b` (7.64 GB compressed). Not pulled; 915 MB free |
