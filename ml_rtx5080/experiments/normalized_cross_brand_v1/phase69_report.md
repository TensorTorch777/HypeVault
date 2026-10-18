# Phase 69: cross-brand generalization on surface-normalized images

Question: does the Phase 45 leave-one-brand-out result survive when both folder classes share one size, encoder, and metadata state?

Answer on this catalog: yes. All five held-out brands stay above 0.998 F1 after normalization, and the pre-specified STRONG band still holds. The claim is real photograph versus these AI-generated images. It is not authentic versus counterfeit watch, and it does not cover other image generators.

Scope and numbering: this phase was run on the project owner's direct instruction of 18 October 2026. It studies real photographs versus AI-generated images under the historical folder labels and makes no authenticity (counterfeit) claim, so `DATA_READINESS = NO_GO_FOR_AUTHENTICITY_CLAIMS` and the Phase 55–68 acquisition gates on `main` are unchanged. It was numbered Phase 56 while it ran; `protocol_v1.json` has no phase number, so its recorded hash is unaffected. It is renumbered Phase 69 because `main` already uses Phase 56 for a different phase.

Protocol `protocol_v1.json`, SHA-256 `ef4936c502f905d4bb70af7dfab62d083e7cb1e36471be3047b349e556d3ddf4`. Fold manifest SHA-256 `55ff5290ff6cc97f773349d93efb8a50a09dc95af1a4a4011fa66a6d627860f5`. The locked final test was not opened. The frozen `epoch_018.pt` was not loaded. Every model started from the timm pretrained `vit_base_patch16_dinov3.lvd1689m` backbone.

## Why this phase exists

Phase 48 showed that the rule "512×512 means Label_1" agrees with the folder label on 26,993 of 26,994 non-test images. Every Label_1 file is a baseline JPEG at IJG quality 95 with no EXIF. Label_0 files vary in size, format, quality, and progressive encoding. The conservative eval transform also pads non-square images with grey, and only Label_0 files are non-square. Phase 45 could not separate learned image content from these surface cues.

## Normalization

`normalize_surface.py` rewrites every Phase 45 non-test row: decode, EXIF transpose, flatten alpha on white, center crop to the short side, Lanczos downscale to 448×448, baseline 4:2:0 JPEG at quality 90, no EXIF, no ICC. Both classes are downscaled. Nothing is upscaled. The 160 rows with a short side below 448 are excluded; all 160 are Label_0.

After normalization (`surface_audit.json`): both classes are 448×448 RGB baseline JPEG with one shared quantization table, zero EXIF, zero ICC. The trainer's eval transform is then a uniform 448→512 resize with no padding for either class. File size remains the only one-number separator, ROC-AUC 0.678 (median 48.1 KB Label_0, 60.0 KB Label_1). File size at a fixed quality tracks pixel detail, so it is a content property, not a container field.

Roles, duplicate quarantine, model, hyperparameters, checkpoint selection, calibration, and threshold 0.50 are Phase 45's, unchanged (`cross_brand_train.train_fold`). Only the image files differ.

## Five folds

Counts at threshold 0.50. "AI called real" is the false-authentic count over held-out Label_1 images; "real called AI" is the false-fake count over held-out Label_0 images.

| Held-out brand | Epoch | AI called real | Real called AI | F1 | PR-AUC | Phase 45 AI called real | Phase 45 real called AI |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. Lange & Söhne | 11 | 0 / 2700 | 9 / 2678 | 0.998336 | 0.997046 | 3 / 2700 | 0 / 2717 |
| Audemars Piguet | 9 | 3 / 2700 | 3 / 2667 | 0.998889 | 0.999995 | 5 / 2700 | 0 / 2691 |
| Patek Philippe | 4 | 1 / 2700 | 5 / 2663 | 0.998890 | 0.997968 | 0 / 2700 | 9 / 2684 |
| Richard Mille | 8 | 5 / 2700 | 2 / 2656 | 0.998703 | 0.999256 | 20 / 2700 | 0 / 2702 |
| Vacheron Constantin | 8 | 0 / 2700 | 1 / 2670 | 0.999815 | 1.000000 | 0 / 2700 | 0 / 2700 |

Pooled over the five held-out brands: AI called real 9 / 13,500 (0.067%, Clopper-Pearson 95% 0.030%–0.127%); real called AI 20 / 13,334 (0.150%, 95% 0.092%–0.232%). These intervals treat images as independent. Several images can show the same physical watch, so the true uncertainty is wider.

Means: unseen PR-AUC 0.998853, unseen F1 0.998927, unseen false-authentic rate 0.000667, in-domain minus unseen PR-AUC gap 0.001147. STRONG band thresholds are 0.90, 0.05, 0.05, and a worst fold of 0.10; the worst fold false-authentic rate is 0.001852 (Richard Mille). Result: STRONG.

Normalization moved errors from "AI called real" toward "real called AI": 28 → 9 and 9 → 20 against Phase 45. Richard Mille, the worst Phase 45 fold, improved from 20 to 5.

## Random-split control

Brand-mixed split on the same normalized files. Selected epoch 4, temperature 0.2368. Test: AI called real 1 / 1350, real called AI 0 / 1327, F1 0.999629, PR-AUC 1.0. The brand-disjoint folds sit within about 0.001 F1 of the random split.

## Feature space

No OOD threshold was fit. Richard Mille again has the clearest shift: 29.4% of its held-out images sit above the training Mahalanobis p95 (Phase 45: 29.3%). The other brands are 3.6%–9.0%. The shift is brand appearance, which normalization did not change, and the classifier still separates the classes on that brand.

## Caveats

- Patek Philippe's calibration split was perfectly separated, so the fitted temperature collapsed to 1e-5 and its probabilities are saturated at 0 or 1. The 0.50 decisions and counts do not depend on temperature. Its PR-AUC and ECE are computed on tied scores and should not be compared across folds.
- Every Label_1 file is scaled by one factor (512→448) while Label_0 factors vary. Downscaling with Lanczos then re-encoding leaves weak periodic traces, but this was not measured.
- Prompt style, background, and framing can still differ between folders. This phase removes container and geometry cues, not content-style cues.
- Folder labels are not verified ground truth. Label_1 images come from one generation pipeline; no unseen-generator test was run.
- The CLS-only ablation and DINOv2 were not run.

## Files

`protocol_v1.json`, `surface_audit.json`, `fold_manifest.json`, `fold_*/metrics.json`, `random_split_control.json`, `cross_brand_metrics.json`, `cross_brand_summary.csv`, `train.log`. Weights are in `/dev/shm/hypevault_normalized_v1/` and do not survive a reboot. Normalized images are in `data_normalized_v1/`, which is git-ignored.
