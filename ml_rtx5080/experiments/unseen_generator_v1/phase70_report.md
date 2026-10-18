# Phase 70: detection of images from unseen generators

Question: do the Phase 69 models and the frozen production model detect generated watch images from generators that did not produce `Label_1_Watches`?

Answer: partly. SDXL images are detected at 90.6% by the Phase 69 brand-mixed model and SD 1.5 images at 52.4%, against 99.9% on the training generator in Phase 69. Ranking survives better than the threshold does: against real photos, ROC-AUC is 0.999 for SDXL and 0.986 for SD 1.5. Many new-generator images score above every real control photo but below the 0.50 decision boundary.

Scope and numbering: this phase was run on the project owner's direct instruction of 18 October 2026. It studies real photographs versus AI-generated images under the historical folder labels and makes no authenticity (counterfeit) claim, so `DATA_READINESS = NO_GO_FOR_AUTHENTICITY_CLAIMS` and the Phase 55–68 acquisition gates on `main` are unchanged. It was numbered Phase 57 (and Phase 56 for its models) while it ran; `protocol_v1.json` keeps that wording so its recorded hash still matches. It is renumbered Phase 70 (models from Phase 69) because `main` already uses Phase 57 (and Phase 56 for its models) for a different phase.

Protocol `protocol_v1.json` was written before generation. The frozen checkpoint hash `5a38c93f…` was checked before scoring. The locked final test was not opened. The generator behind `Label_1_Watches` is not recorded. Every Label_1 file is 512×512, so SDXL (native 1024×1024) is the primary unseen generator. SD 1.5 (native 512×512) is reported beside it. Its low detection rate makes it unlikely to be the training generator, but that is not established.

## Images

500 per generator: 5 brands × 5 model lines × 10 scenes × 2 seeds. The prompt is `photo of a {brand} {model line} luxury wristwatch, {scene}` with one fixed negative prompt. SDXL base 1.0 used 1024×1024, 30 steps, guidance 6.0. SD 1.5 used 512×512, 30 steps, guidance 7.5. Raw output is PNG. Each image is also normalized with the Phase 69 rule (448×448, baseline JPEG q90, no metadata). Generation took 44 min (SDXL, 0.19 img/s) and 8 min (SD 1.5, 1.01 img/s) on the RTX 5080.

## Detection at P(generated) ≥ 0.50

Clopper-Pearson 95% intervals in brackets. Images from one prompt are not independent, so the true uncertainty is wider.

| Scorer | Input | SDXL | SD 1.5 |
|---|---|---:|---:|
| Phase 69 random split (primary) | normalized | 453/500, 90.6% [87.7, 93.0] | 262/500, 52.4% [47.9, 56.9] |
| Phase 69 fold, brand unseen | normalized | 452/500, 90.4% [87.5, 92.8] | 290/500, 58.0% [53.5, 62.4] |
| Frozen epoch_018 | raw PNG | 481/500, 96.2% [94.1, 97.7] | 402/500, 80.4% [76.6, 83.8] |
| Frozen epoch_018 | normalized | 497/500, 99.4% [98.3, 99.9] | 433/500, 86.6% [83.3, 89.5] |

Phase 69 random split, per brand (SDXL / SD 1.5): Lange 88 / 41, Audemars Piguet 90 / 69, Patek Philippe 94 / 49, Richard Mille 90 / 58, Vacheron Constantin 91 / 45, each out of 100.

## Real-photo control

A detection rate alone cannot separate a good detector from one that calls everything generated. Each scorer also scored 500 real photos (100 per brand) that it never trained on, through the same input path.

| Scorer | Control source | Real called generated | AUC vs SDXL | AUC vs SD 1.5 | TPR at 1% FPR, SDXL / SD 1.5 |
|---|---|---:|---:|---:|---:|
| Phase 69 random split | Phase 69 random_test, normalized | 0/500 [0, 0.74%] | 0.9991 | 0.9858 | 0.986 / 0.846 |
| Phase 69 fold | held-out brand's cross_brand_test, normalized | 0/500 [0, 0.74%] | 0.9872 | 0.9423 | 0.942 / 0.788 |
| Frozen epoch_018, raw | split v2 validation, original files | 0/500 [0, 0.74%] | 0.9984 | 0.9766 | 0.994 / 0.940 |
| Frozen epoch_018, normalized | the same rows, normalized | 2/500 [0.05, 1.44%] | 0.9981 | 0.9943 | 0.998 / 0.958 |

The frozen control rows were used for epoch selection, not for training. TPR at 1% FPR places the threshold at the 99th percentile of that control set. It describes ranking quality and is not a deployable threshold, because it was read from the same control photos.

For the Phase 69 random-split model, the highest real-photo logit is −3.51 and the median SD 1.5 logit is 0.76. Images with logits between those values rank as more generated than every real control photo but are still called real at 0.50. The miss is mostly calibration to the training generator, not loss of separation.

## Reading

- New generators reduce detection. SD 1.5 roughly halves it for the normalized model. The Phase 69 cross-brand result does not extend to new generators.
- The frozen production model, trained on un-normalized images, detects more new-generator images than the normalized model. The 512×512 size cue cannot explain this, because the frozen model detects SDXL best after normalization (448 px) and its raw SDXL input is 1024 px. One possibility is that 512→448 downscaling during normalization removes high-frequency traces the frozen model uses. That was not tested.
- Moving the threshold could recover most SD 1.5 detections, but only a threshold fixed on separate data counts. The next step is a held-out new-generator set for threshold selection plus another for testing. Adding SDXL and SD 1.5 images to training would be a stronger fix.

## Not claimed

Authentic versus counterfeit watches. Generators other than SDXL base 1.0 and SD 1.5. Images from other prompt styles. Independence between images that share a prompt.

## Files

`protocol_v1.json`, `generation_manifest.csv` (seed and prompt per image), `scores.csv` (four logits per image), `detection_metrics.json`, `control_scores.csv`, `control_metrics.json`, `operating_points.json`, `generate.log`. Images are in `data_unseen_generator_v1/` (git-ignored). Script `ml_rtx5080/unseen_generator.py`; tests `tests/test_unseen_generator.py`.
