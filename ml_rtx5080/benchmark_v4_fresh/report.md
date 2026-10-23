# Benchmark v4 (fresh) — Phase 74 "DINOv3 Kandinsky v4" final test

Run `20261022T204338Z`, scored once. Protocol `protocol.json` (SHA-256 `a313d340…`), written before any v4 data. Rule frozen before scoring: `experiments/kandinsky_v4/fusion_rule.json` (`d1201673…`).

## Phase 74 system

One new linear head on the live fine-tuned DINOv3 (`epoch_018`, hash-checked, FP32). It reads the two vectors the existing head already computes: the CLS token and the attention-pooled patch embedding. It needs one forward pass and no extra model.

- **Training data (development only):** 5,000 Label_0 photos and 900 benchmark_v1–v3 eBay photos as real. As generated: Kandinsky 2.2 (852 from benchmark_v1–v3 plus 534 new images, seeds base 78000, scenes T1–T8), SDXL, aMUSEd and Stable Cascade lite from benchmark_v1–v3.
- **Thresholds from real photos only:** 1,000 Label_0 validation photos plus the 900 eBay photos scored out-of-fold. At most 1% may be called FAKE, and at most 5% more go to REVIEW. DINOv3's own REVIEW gate is kept.

### How it was chosen (development data, each benchmark version held out in turn)

| Strategy | Kandinsky caught | Real FAKE |
|---|---|---|
| AEROBLADE-style MoVQ reconstruction error (training-free; Ricker et al., CVPR 2024) | 23.8–29.5% | 10 / 900 |
| Aligned MoVQ reconstructions only (Rajan et al., ICLR 2025) | 26.3–45.8% | 15–19 / 900 |
| Few-shot heads on pretrained / fine-tuned DINOv3 (Cozzolino et al., CVPRW 2024) | 100% | 15–16 / 900 |
| **Selected: fine-tuned DINOv3 features, several generators, eBay reals** | **852 / 852 (100%)** | **11 / 900** |

Scene-template check: trained on T1–T4 scenes, the selected head caught 100% of T5–T8 Kandinsky. Trained on T5–T8, it caught 97.9–100% of T1–T4.

## Data (512 × 512 JPEG q90)

| Group | Images |
|---|---|
| Real eBay photos (new listings; Richard Mille at the 30 minimum) | 270 |
| **Kandinsky 2.2, seeds base 76000 (primary)**: 140 from development scenes T1–T8, 143 from new scenes T9–T16 | 283 |
| Stable Cascade lite, seeds base 77000 (secondary) | 297 |

## Results (FAKE decisions; REVIEW reported separately)

| System | Real FAKE | Real REVIEW | Kandinsky caught | of which T9–T16 (new scenes) | Stable Cascade caught |
|---|---|---|---|---|---|
| DINOv3 alone | 0 / 270 (0.0%) | 25 | 23 / 283 (8.1%) | 17 / 143 | 203 / 297 (68.4%) |
| Watch forensics v2 (Phase 73) | 1 / 270 (0.4%) | 34 | 136 / 283 (48.1%) | 87 / 143 | 279 / 297 (93.9%) |
| **DINOv3 Kandinsky v4 (Phase 74)** | **2 / 270 (0.7%, CI 0.1–2.7%)** | 34 | **283 / 283 (100%, CI 98.7–100%)** | **143 / 143** | **297 / 297 (100%)** |

**Paired, primary set (165 Kandinsky images with two binary decisions):** v4 caught 142 that DINOv3 alone missed, and DINOv3 alone caught 0 that v4 missed. Exact McNemar p = 3.6 × 10⁻⁴³.

## Pre-registered success criterion: **met**

- At least 80% of Kandinsky v4 images called FAKE: met (100%).
- At most 3% of real photos called FAKE: met (0.7%).

## After scoring: does the recipe work on a generator it has never seen?

This check was not pre-registered. It ran after benchmark_v4 on development data only (`kandinsky_v4_design.py unseen`, `experiments/kandinsky_v4/leave_one_generator_out.json`). Each time, one generator is removed from training entirely, and the head is trained on the rest. For Kandinsky, that means none of the 1,386 Kandinsky images. The threshold comes from real photos only, as above.

| Generator never seen in training | Fine-tuned DINOv3 features (selected) | Pretrained DINOv3 features | Both |
|---|---|---|---|
| **Kandinsky 2.2** | **1,154 / 1,386 (83.3%)** | 1,183 / 1,386 (85.4%) | 1,294 / 1,386 (93.4%) |
| aMUSEd | 292 / 293 (99.7%) | 290 / 293 (99.0%) | 291 / 293 (99.3%) |
| SDXL | 148 / 150 (98.7%) | 97 / 150 (64.7%) | 147 / 150 (98.0%) |
| Stable Cascade lite | 293 / 296 (99.0%) | 289 / 296 (97.6%) | 292 / 296 (98.6%) |

eBay real photos called FAKE (out-of-fold) were 0.9–1.6%. Training on several different generators carries over to one the head has not seen. That is what the 100% on benchmark_v4 rests on, not memorising Kandinsky. It is still four generators from a similar era with the same watch prompts. A pre-registered test on a newer unseen generator (for example FLUX or SD3) is the next step.

## Limits

- **Kandinsky 2.2 and Stable Cascade are seen generators now.** Their images were in training. This is not an unseen-generator result. The disk had 7 GB free, so no new generator could be downloaded for an unseen check.
- On development data, the head caught most images from generators held out of training (Stable Cascade 294/296 when v3 was held out). That is one held-out generator per fold, not a fresh test.
- 13% of real photos go to REVIEW (34 / 270), mostly through DINOv3's own REVIEW gate. That is a human-review workload.
- Single run, 270 real photos. The real-photo FAKE rate has a wide interval (0.7%, 95% CI 0.1–2.7%).
- The prompts are templated watch scenes. Real-world generated listings may differ in content and post-processing.
