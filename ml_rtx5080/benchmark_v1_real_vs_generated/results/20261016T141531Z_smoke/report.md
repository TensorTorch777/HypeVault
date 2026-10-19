# Benchmark v1 — real photos vs generated images (SMOKE_TEST)

Run `20261016T141531Z_smoke`, protocol SHA-256 `9206017181261f29…`. Sample minimums were not met. These numbers check the pipeline and may not be reported as results.

## Secondary: REAL_PHOTO + sdxl_base10 (10 images)

| Model | Coverage | Accuracy | Balanced acc. | False-authentic (upper 95%) | False-fake (upper 95%) | ROC-AUC |
|---|---|---|---|---|---|---|
| dinov2_legacy | 100.0% | 0.0% | n/a | 10/10 (100.0%) | 0/0 (n/a) | n/a |
| dinov3_experimental | 0.0% | n/a | n/a | 0/10 (25.9%) | 0/0 (n/a) | n/a |

Paired: 0 images with two binary decisions, 0 discordant (DINOv3 only correct 0, DINOv2 only correct 0), exact McNemar p = 1.0000.

