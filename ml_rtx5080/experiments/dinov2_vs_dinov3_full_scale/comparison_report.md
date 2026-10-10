# DINOv2 vs DINOv3 full-scale frozen comparison

DATA_READINESS = NO_GO_FOR_AUTHENTICITY_CLAIMS

No model is selected. Directory labels are historical dataset labels, not verified authenticity. The common held-out validation cohort is 315 authentic-labeled A. Lange & Söhne images and 0 fake-labeled images. Ranking metrics on that cohort are undefined. The stored manifests do not match a Vacheron Constantin versus Richard Mille validation split, and no other subset was promoted to validation.

## What was compared

The live DINOv2 checkpoint `fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66` uses 504-square ImageNet preprocessing and the 0.50 / 0.88 operational policy. Its sigmoid is uncalibrated. The experimental DINOv3 checkpoint `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f` uses `resize_pad_square_eval_v1` at 512, temperature 0.24038200410185356, threshold 0.50, and the shadow quality review rule. Temperature-scaled scores are not on the DINOv2 scale. Both checkpoints were hashed before and after the sweep. Neither weight file was written.

The models were not trained on the same images. DINOv2's stored split is a product-folder split that also contains sneakers. DINOv3 uses the brand-stratified watch manifest. Images in either training or calibration membership are diagnostic only.

## Primary paired cohort

315 images, both models successful on 315, agreement 281, disagreement 34, both binary and both wrong 0.

| Model | Successful | Failed | AUTHENTIC-labeled | FAKE-labeled | REVIEW | Coverage | Accuracy | Precision | Recall | F1 | Balanced accuracy | False-authentic rate | False-fake rate | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dinov2_legacy | 315 | 0 | 315 | 0 | 0 | 1 | 0.996825 | 0 | undefined | undefined | undefined | undefined | 0.0031746 | undefined | undefined |
| dinov3_experimental | 315 | 0 | 315 | 0 | 33 | 0.895238 | 1 | undefined | undefined | undefined | undefined | undefined | 0 | undefined | undefined |

False-authentic and false-fake rates use every successful image of that inherited label as the denominator. REVIEW stays in that denominator and out of the binary accuracy denominator. DINOv2 has no REVIEW state.

## Diagnostic sweep

Train and calibration outputs are not validation. DINOv3's own validation still contains 2682 images that are in the live DINOv2 training folders, so that split is not a paired held-out comparison.

| Model | Successful | Failed | AUTHENTIC-labeled | FAKE-labeled | REVIEW | Coverage | Accuracy | Precision | Recall | F1 | Balanced accuracy | False-authentic rate | False-fake rate | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dinov2_legacy | 26679 | 0 | 13179 | 13500 | 0 | 1 | 0.986656 | 0.984446 | 0.989259 | 0.986847 | 0.986624 | 0.0107407 | 0.0160103 | 0.998607 | 0.998894 |
| dinov3_experimental | 26679 | 0 | 13179 | 13500 | 1822 | 0.931707 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 |

## Serving

Parity against the CPU FP32 reference on 24 non-test images passed with max absolute logit error 2.360e-05 for DINOv2 and 6.819e-05 for DINOv3. The tolerance was 1e-4. Triton batches stayed at 8 and 16 on KIND_GPU. Peak sampled GPU memory was 13378.0 MiB. Locked final-test images opened: 0.

Per-image latency is the Triton request wall time divided by the images in that request. Primary-cohort latency is inside the machine-readable metrics. This run does not change production approval, publication, temperature, or the 0.88 floor.
