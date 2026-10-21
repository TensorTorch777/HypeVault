# Benchmark v1 — real photos vs generated images (CONFIRMATORY)

Run `20261020T110244Z`, protocol SHA-256 `9206017181261f29…`. No performance winner, and DINOv3 did not meet the engineering-selection conditions. DINOv2 remains the live model.

## Primary: REAL_PHOTO + heldout_kandinsky22 (584 images)

| Model | Coverage | Accuracy | Balanced acc. | False-authentic (upper 95%) | False-fake (upper 95%) | ROC-AUC |
|---|---|---|---|---|---|---|
| dinov2_legacy | 100.0% | 53.4% | 52.1% | 271/284 (97.3%) | 1/300 (1.6%) | 0.6092 |
| dinov3_experimental | 76.5% | 62.0% | 53.3% | 170/284 (64.7%) | 0/300 (1.0%) | 0.8275 |

Paired: 447 images with two binary decisions, 16 discordant (DINOv3 only correct 11, DINOv2 only correct 5), exact McNemar p = 0.2101.

## Secondary: REAL_PHOTO + sdxl_base10 (450 images)

| Model | Coverage | Accuracy | Balanced acc. | False-authentic (upper 95%) | False-fake (upper 95%) | ROC-AUC |
|---|---|---|---|---|---|---|
| dinov2_legacy | 100.0% | 82.0% | 73.2% | 80/150 (60.3%) | 1/300 (1.6%) | 0.8593 |
| dinov3_experimental | 90.4% | 94.3% | 91.9% | 23/150 (21.0%) | 0/300 (1.0%) | 0.9805 |

Paired: 407 images with two binary decisions, 59 discordant (DINOv3 only correct 56, DINOv2 only correct 3), exact McNemar p = 0.0000.

