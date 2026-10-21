# Benchmark v2 (fresh) — Phase 72 "watch forensics" final test

Run `20261020T154603Z`, scored once. Protocol `protocol.json` (SHA-256 `bd1a4487…`), written before any v2 data. Fusion rule `experiments/watch_forensics_v1/fusion_rule.json` (SHA-256 `9f37f328…`), frozen on benchmark_v1 development data before any v2 image was scored.

## Data (512 × 512 JPEG q90 for every image)

| Group | Images | Notes |
|---|---|---|
| Real eBay photos | 300 | New listings, none in benchmark_v1. 166 more were excluded as near-copies of training images |
| Kandinsky 2.2, fresh seeds | 287 | Same generator as v1, new images |
| aMUSEd 512 | 293 | Masked-token transformer + VQ-GAN; not diffusion; never used by any HypeVault model |

## Results (FAKE decisions; REVIEW counted separately)

| System | Real rejected | Real REVIEW | Kandinsky caught | aMUSEd caught | All generated caught |
|---|---|---|---|---|---|
| DINOv3 alone (fine-tuned) | 1 / 300 (0.3%) | 40 | 17 / 287 (5.9%) | 123 / 293 (42.0%) | 140 / 580 (24.1%) |
| DINOv2 alone (live) | 3 / 300 (1.0%) | 0 | 21 / 287 (7.3%) | 109 / 293 (37.2%) | 130 / 580 (22.4%) |
| Phase 71 probe | 6 / 300 (2.0%) | 0 | 35 / 287 (12.2%) | 276 / 293 (94.2%) | 311 / 580 (53.6%) |
| **Watch forensics (fusion A)** | **11 / 300 (3.7%, CI 1.8–6.5%)** | 40 | **79 / 287 (27.5%)** | **289 / 293 (98.6%)** | **368 / 580 (63.4%, CI 59.4–67.4%)** |

**Paired, generated images with two binary decisions (315):** watch forensics caught 54 that DINOv3 missed, and DINOv3 caught 0 that it missed. Exact McNemar p = 1.1 × 10⁻¹⁶.

## Against the pre-registered success criterion

- More generated images caught than DINOv3 alone, McNemar p < 0.05: **met** (63.4% vs 24.1%).
- Real-photo rejection at most 3 percentage points above DINOv3 alone: **not met**. The fusion rejected 3.7% of real photos and DINOv3 0.3%, a difference of 3.3 points. One fewer rejected real photo would have met it.
- **Overall: the pre-registered success criterion was not met.** The rule is not changed after seeing these results.

## What it shows

- **The decoder-trace idea generalises across generator families.** The Phase 71 probe, trained only on SD 1.5 and SDXL autoencoder copies of real photos, caught 94.2% of aMUSEd images. aMUSEd uses a completely different decoder (VQ-GAN). The fine-tuned DINOv3 caught 42.0%.
- **Brand-text forensics adds independent evidence.** On fresh Kandinsky images, the fusion caught 27.5%, against 12.2% for the probe alone and 5.9% for DINOv3 alone.
- **Kandinsky 2.2 is still the hard case:** 72.5% are not caught (87 go to REVIEW).
- **Cost:** 11 of 300 real photos rejected (3.7%), against 1 for DINOv3 alone.
