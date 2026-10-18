# Phase 34 production-readiness audit

Classification: **CONDITIONALLY_READY**.

This audit does not promote the policy and does not open final-test images.

## Reasons

- promotion requirements not met: real_ood_validation_complete, production_runtime_parity_complete, production_serving_infrastructure_available, quality_review_workload_accepted, policy_formally_approved
- in-distribution model evidence and the PyTorch shadow path are intact; remaining blockers are infrastructure, data, or policy approval

## Identity

- checkpoint_sha256: `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`
- membership_hash: `78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678`
- manifest_file_sha256: `4d0d20c17d12ac29cb299b7b4f369ebcce3c5acd7b2fd5430d61f952b9852eea`
- temperature: `0.24038200410185356`
- policy_version: `shadow_v1`
- current_production_status: `not_promoted`

## Model and calibration

- Locked final test: 3006 samples, ROC-AUC 1.0, PR-AUC 1.0, F1 1.0, FAR 0.0, false-fake rate 0.0.
- Brands: A. Lange & Söhne, Audemars Piguet, Patek Philippe, Richard Mille, Vacheron Constantin.
- Calibration samples: 2998.
- Calibrated NLL 2.5221722488311116e-09, Brier 8.573122380084356e-18, ECE 0.0.

## Robustness and quality

- Clean reference pool false-authentic 0, false-fake 0.
- Stress coverage 8/9. Stress false-authentic 0.
- known_robustness_limitation: `resize_recompression_case`.
- Review rate 0.06871897458694524 (1855 / 26994). Authentic review rate 0.1168667555950793. Fake review rate 0.020592592592592593.

## OOD and serving

- in_distribution_validation: complete.
- open_set_validation: incomplete. The classifier is not open-set protected.
- SHADOW_READY: True. PRODUCTION_RUNTIME_READY: False.
- Batch-1 FP32 model p50: 13.870440496248193 ms.
- Batch-4 throughput: 82.6969435695549 images/sec.
- Mixed-corpus end-to-end mean before the fast gate: 75.96065286453813 ms. Median 31.340526504209265 ms.
- 2560 quality-only fast-gate p50: 76.33416149474215 ms.
- Fast-gate end-to-end p50, decode included and disk I/O excluded: 550 px 23.975707503268495 ms; 2560 px 214.6484240074642 ms.
- Memory growth: 0 bytes.

## Hypothetical review volume

- 100 requests/day -> 6.8719 REVIEW. No SLA is assumed.
- 1000 requests/day -> 68.7190 REVIEW. No SLA is assumed.
- 10000 requests/day -> 687.1897 REVIEW. No SLA is assumed.

## Residual risks

- R1 Resize/recompression false-fake (high, open): One authentic stress image remains FAKE.
- R2 OOD unvalidated (high, open): The system is not open-set protected.
- R3 Triton unavailable (high, open): No validated Triton production runtime.
- R4 TensorRT unavailable (high, open): No validated TensorRT runtime.
- R5 ONNX CUDA unavailable (high, open): No validated GPU ONNX runtime.
- R6 Quality review workload (medium, open): Operations has not accepted the review burden.
- R7 Five-brand dataset limitation (medium, residual): Results do not establish universal luxury-watch performance.
- R8 Missing product/SKU metadata (medium, residual): Product-disjoint evaluation is unavailable.
- R9 Missing marketplace/source metadata (medium, residual): Source shift cannot be measured.
- R10 Live backend remains DINOv2 (high, open): Promotion has not happened. The live model is a different stack.

Severity words: critical means a promoted system would fail open or contradict the locked safety result; high means a required control is missing; medium means a scope or operations limit.

final_test_samples_used = 0.
