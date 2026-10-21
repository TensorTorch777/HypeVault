# Benchmark v3 (fresh) — Phase 73 "watch forensics v2" final test

Run `20261020T203323Z`, scored once. Protocol `protocol.json` (SHA-256 `a2b3cc80…`), written before any v3 data. Rules frozen before scoring:
- `watch_forensics_v1/fusion_rule.json` (`9f37f328…`, Phase 72 fusion A);
- `watch_forensics_v2/fusion_rule.json` (`2bb10caf…`, Phase 73).

## Phase 73 changes (fixes 1–5), designed on development data only

1. **Thresholds from real photos only:** 600 eBay photos (out-of-fold) plus 1,000 Label_0 validation photos. At most 1% may be called FAKE.
2. **REVIEW tier:** at most 5% more real photos go to REVIEW by score.
3. **Dial reading:** a centre crop at double resolution and docTR as a second literal reader. Brand text found on 90% / 79% / 77% of real development photos, up from 80% / 67% / 69%.
4. **Probe retrained on five decoders:** SD 1.5, SDXL, aMUSEd VQ-GAN, TAESD, TAESDXL.
5. **Learned combiner:** logistic regression on DINOv3 logit, DINOv3 REVIEW, probe logit and brand-text readers.

## Data (512 × 512 JPEG q90)

| Group | Images |
|---|---|
| Real eBay photos (new listings) | 300 |
| **Stable Cascade lite (primary; Würstchen with Paella VQ-GAN, unseen by every component)** | 296 |
| Kandinsky 2.2, new seeds (secondary; a development generator) | 281 |

## Results (FAKE decisions; REVIEW reported separately)

| System | Real FAKE | Real REVIEW | Stable Cascade caught | Kandinsky caught |
|---|---|---|---|---|
| DINOv3 alone | 1 / 300 (0.3%) | 41 | 204 / 296 (68.9%) | 8 / 281 (2.8%) |
| DINOv2 alone | 2 / 300 (0.7%) | 0 | 164 / 296 (55.4%) | 29 / 281 (10.3%) |
| Watch forensics v1 (Phase 72) | 9 / 300 (3.0%) | 41 | 259 / 296 (87.5%) | 66 / 281 (23.5%) |
| **Watch forensics v2 (Phase 73)** | **6 / 300 (2.0%)** | 45 | **272 / 296 (91.9%)** | **79 / 281 (28.1%)** |

**Paired, primary set (242 images with two binary decisions):** v2 caught 33 that DINOv3 missed, and DINOv3 caught 0 that v2 missed. Exact McNemar p = 2.3 × 10⁻¹⁰.

## Pre-registered success criterion: **met**

- More Stable Cascade images caught than DINOv3 alone, p < 0.05: met (91.9% vs 68.9%).
- Real-photo FAKE rate at most 3 points above DINOv3 alone: met (+1.7 points).

## Limits

- 15% of real photos go to REVIEW (45 / 300), mostly through DINOv3's own REVIEW gate. That is a human-review workload.
- Kandinsky 2.2 is still mostly missed (28.1% caught, 40% to REVIEW).
- Single run, 300 real photos per benchmark. Real-photo rates have wide intervals (2.0%, 95% CI 0.7–4.3%).
