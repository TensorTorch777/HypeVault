# Phase 45 cross-brand report

Question: can DINOv3 learn authenticity cues that generalize to a luxury-watch brand completely unseen during training?

Answer on this catalog: yes, under the pre-specified STRONG band. The five leave-one-brand-out folds stay at the ceiling. This does not extend the claim to brands outside the five-brand audited set.

Protocol hash `06be2fbbfc1761b2655ac81657f63c863bfeb02d9c016a57c85ec1964a1eb80d`. Manifest hash `b5f7ad5f0906d18a87d75576a7cff8fbf1844ff0428583461037c0825a9ed2a9`. Both match the recorded files. Frozen checkpoint `epoch_018.pt` hash `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f` matches and was not used for initialization. Models started from the timm pretrained `vit_base_patch16_dinov3.lvd1689m` backbone. The locked final-test manifest was not opened for scoring, selection, or calibration.

## Five primary folds

Head: `cls_patch_attention`. Input 512. Training BF16. Evaluation FP32. Checkpoint selection used only the four training brands' validation data: maximum validation PR-AUC subject to false-authentic rate at most 0.02, then higher F1, lower false-authentic rate, lower validation loss, earlier epoch. Threshold fixed at 0.50. Temperature was fit on the four-brand calibration split and frozen before the held-out brand was scored.

| Held-out brand | Epoch | Temperature | In-domain PR-AUC | Unseen PR-AUC | In-domain FAR | Unseen FAR | Unseen F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. Lange & Söhne | 6 | 0.542368 | 1.000000 | 1.000000 | 0 | 0.001111 | 0.999444 |
| Audemars Piguet | 9 | 0.230932 | 1.000000 | 1.000000 | 0 | 0.001852 | 0.999073 |
| Patek Philippe | 2 | 0.499293 | 1.000000 | 1.000000 | 0 | 0 | 0.998336 |
| Richard Mille | 4 | 0.526151 | 1.000000 | 0.999985 | 0 | 0.007407 | 0.996283 |
| Vacheron Constantin | 15 | 0.240419 | 1.000000 | 1.000000 | 0 | 0 | 1.000000 |

Unseen false-authentic counts, all over 2700 fakes: Lange 3, Audemars Piguet 5, Patek Philippe 0, Richard Mille 20, Vacheron Constantin 0. Patek Philippe's unseen false-fake rate is 0.003353 (9 of 2684 authentic images scored as fake). Every other unseen false-fake rate is 0.

Ranking by unseen false-authentic rate, then PR-AUC, F1, and ECE: Vacheron Constantin, Patek Philippe, A. Lange & Söhne, Audemars Piguet, Richard Mille.

## Random-split control

Same architecture, brand-mixed split, not the locked final test. Validation n=2701 and random test n=2696. Both have ROC-AUC 1, PR-AUC 1, F1 1, false-authentic rate 0, false-fake rate 0. Selected epoch 2. Frozen temperature 0.032715. The random split does not hide a failure that the brand-disjoint split reveals. Both protocols sit on the ceiling.

## CLS-only ablation

Identical manifests and training rules. Unseen false-authentic rates: Lange 0, Audemars Piguet 0.001852, Patek Philippe 0, Richard Mille 0.007037, Vacheron Constantin 0. Unseen PR-AUC stays at or above 0.999517. Patch attention does not improve the worst brand's false-authentic rate. Richard Mille is 0.007407 with patch attention and 0.007037 with CLS-only.

## Feature space

No OOD threshold was fit. Richard Mille is the only held-out brand with a clear shift: 29.3% of its images sit above the training Mahalanobis p95, against about 5% expected for the same distribution. Mean P(fake) on that distant slice is 0.000177, so the head still calls those images authentic. The other four brands have 3.2% to 8.3% of images above the training p95.

## Classification

Pre-specified STRONG rule: mean unseen PR-AUC at least 0.90, mean unseen false-authentic rate at most 0.05, mean PR-AUC gap at most 0.05, and no fold unseen false-authentic rate above 0.10. Measured values: 0.999997, 0.002074, 0.000003, and a worst fold of 0.007407. Result: STRONG inside this five-brand catalog.

DINOv2 ViT-B/14 was not run. Production routes, the publication gate, and the frozen candidate were not changed.
