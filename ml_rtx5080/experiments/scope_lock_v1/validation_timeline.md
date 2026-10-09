# Validation timeline

Phases 16–33 follow the design sequence in `new_architecture.md` where an executed artifact exists. Phases 34–38 are the audits after the checkpoint was frozen. None of these phases changed `epoch_018.pt` after the final-test freeze, and none changed production status to promoted.

| Phase | Objective | Major result | Artifact | Frozen model changed | Production status changed |
|---|---|---|---|---|---|
| 16 | Lock the audited split used by later calibration. | 30,000 images, membership hash `78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678`. No product or marketplace identity. | `ml_rtx5080/experiments/dataset_audit/split_manifest_v2.json` | No | No |
| 17 | Test whether a real OOD corpus already existed. | `REAL_OOD_CORPUS_AVAILABLE = false`, 0 samples. | `ml_rtx5080/experiments/ood_v1/ood_summary.json` | No | No |
| 18 | Mine hard negatives inside the catalog. | Challenge manifest exists. It is not an open-set corpus. | `ml_rtx5080/experiments/hard_negative_v1/challenge_manifest.json` | No | No |
| 19 | Compare DINOv2 and DINOv3 heads. | The frozen candidate is DINOv3 `cls_patch_attention`. | `ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/config.json` | No, after freeze | No |
| 20 | Fit one temperature on the Phase 16 calibration split. | Temperature `0.24038200410185356` on 2,998 samples. | `ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/calibration/calibration_summary.json` | No | No |
| 21 | Score the locked final test once. | 3,006 samples, calibrated ROC-AUC, PR-AUC, and F1 of 1.0, false-authentic 0, false-fake 0. | `ml_rtx5080/experiments/final_test_v1/final_test_summary.json` | No | No |
| 22 | Multi-image authentication design. | Design text only. No promotion result. | `new_architecture.md` | No | No |
| 23 | Image-quality gate and stress audit. | 8/9 known stress false-fake events caught. Resize/recompression of one Richard Mille file remains uncovered. | `ml_rtx5080/experiments/image_quality_v2/quality_operating_points.json` | No | No |
| 24 | Shared evaluation preprocessing. | Evaluation transform `resize_pad_square_eval_v1` at 512. | `ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/config.json` | No | No |
| 25 | Record a model registry without switching the live path. | Live customer inference stayed DINOv2. | `backend/inference/dinov2_model.py` | No | No |
| 26 | ONNX parity. | CUDA ONNX is not the authoritative path. | `ml_rtx5080/experiments/production_readiness_v1/readiness_report.json` | No | No |
| 27 | TensorRT benchmark plan. | TensorRT is unavailable for this candidate. | `ml_rtx5080/experiments/production_readiness_v1/readiness_report.json` | No | No |
| 28 | Triton deployment plan. | Triton is not serving the frozen DINOv3 checkpoint. | `ml_rtx5080/experiments/production_readiness_v1/readiness_report.json` | No | No |
| 29 | Backend decision policy. | Shadow policy `shadow_v1` is not the live verdict. | `ml_rtx5080/experiments/policy_shadow_v1/policy_summary.json` | No | No |
| 30 | Observability design. | No promotion from metrics alone. | `new_architecture.md` | No | No |
| 31 | Reproducibility and numeric audit. | BF16 was not made authoritative. | `ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/config.json` | No | No |
| 32 | Regression tests. | The suite remained the check against silent contract changes. | `tests/` | No | No |
| 33 | Final model comparison. | Selected checkpoint remained `epoch_018.pt` with SHA-256 `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`. | `ml_rtx5080/experiments/final_test_v1/final_test_summary.json` | No | No |
| 34 | Production-readiness audit. | `CONDITIONALLY_READY`. Production status `not_promoted`. | `ml_rtx5080/experiments/production_readiness_v1/readiness_report.json` | No | No |
| 35 | Real OOD evaluation. | `OOD_FAIL`. 408 samples, 337 authentic escapes, unseen luxury 0.8814. | `ml_rtx5080/experiments/ood_validation_v1/ood_status.json` | No | No |
| 36 | Independent watch-like OOD. | `WATCH_OOD_UNRESOLVED`. 25 holdout images. | `ml_rtx5080/experiments/watch_like_ood_v1/watch_like_ood_status.json` | No | No |
| 37 | Acquire a protocol-sized original corpus. | `WATCH_OOD_V2_CORPUS_READY = false`. 0 admitted. | `ml_rtx5080/experiments/watch_like_ood_v2/acquisition_status.json` | No | No |
| 38 | Feasibility review. | `SCOPE_REDUCTION_RECOMMENDED`. Governance infeasible. Promotion false. | `ml_rtx5080/experiments/ood_strategy_v1/recommendation.json` | No | No |

Phase 35 is not described here as invalid. Phase 36 is not described here as confirmation.
