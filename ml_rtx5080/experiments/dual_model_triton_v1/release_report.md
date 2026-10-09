# Phase 50 release report

The dual-model selector, research access control, and customer-label safety work are on `feature/dual-model-triton`, and draft PR [#1](https://github.com/TensorTorch777/HypeVault/pull/1) is open. CPU CI passes. The DINOv3 candidate passes pre-registered parity on CPU Triton. GPU serving is blocked, and DINOv2 does not load in the available Triton. The PR is not merged.

## Evidence

- **Checkpoint:** `epoch_018.pt` SHA-256 matches `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`. Temperature remains `0.24038200410185356` and the threshold `0.50`.
- **CI:** run 37914776725 on `7c9501b`. Backend: 50 tests, OK, 2 skips with named local-only dependencies. Frontend: lint 0 warnings, `tsc` passed, `next build` passed.
- **Clean checkout:** the same commands pass from a `git archive` extract with no `.env` files.
- **Local suite:** 428 tests with 3 existing Triton/ONNX/TensorRT skips. One environmental disk-full error was resolved by freeing build cache; the full suite is re-run before the final summary.
- **CPU Triton parity:** `parity_results.json`, protocol `858abd0c…e4cb` committed in `b639dc3` before results. Max absolute logit error `3.08e-5` (tolerance `1e-4`). Max absolute probability error `3.30e-7` (tolerance `1e-5`). 0 decision mismatches at every batch size.
- **End-to-end:** `integration_test_results.json`. All 13 flows pass against the local stack with CPU Triton. DINOv2 flows show fail-closed behaviour only, because the model does not load.
- **Browser:** the research page disables DINOv2 as unavailable, selects DINOv3, and shows `EXPERIMENTAL — NOT APPROVED FOR PRODUCTION`.

## Not done

- GPU Triton serving, GPU parity, latency, and memory. Blocked by the container runtime, root access, and disk space.
- DINOv2 Triton serving and DINOv2 parity. Blocked by ONNX IR 10 on Triton 23.10, plus the owner decision on which DINOv2 backs the selector.
- Merge to `main`. Needs review and approval.
- Deployment. Out of scope.

`DINOv3_PRODUCTION_ALLOWED = false`

`PRODUCTION_PROMOTION_ALLOWED = false`
