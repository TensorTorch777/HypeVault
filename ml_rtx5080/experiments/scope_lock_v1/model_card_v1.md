# Model card v1

## Model identity

- Family: DINOv3 ViT-B/16
- Backbone: `vit_base_patch16_dinov3.lvd1689m`
- Head: `cls_patch_attention`
- Input size: 512
- Authoritative path: PyTorch CUDA FP32. BF16 is not authoritative.
- Checkpoint: `ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/epoch_018.pt`
- Checkpoint SHA-256: `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`
- Temperature: `0.24038200410185356`
- Policy: `shadow_v1`
- Scope version: `scope_lock_v1`

## Intended use

Research prototype for authenticity classification within five evaluated luxury-watch brands:

1. A. Lange & Söhne
2. Audemars Piguet
3. Patek Philippe
4. Richard Mille
5. Vacheron Constantin

The caller must declare one of those brands. The authenticity head is not a brand identifier.

`brand_input_mode` is `USER_DECLARED`. `brand_identity_verified_by_model` and `brand_identity_verified_by_image` are false. The scope gate is `DECLARATION_ONLY`.

The supplied brand is a user-declared scope parameter and is not independently verified from the image. Brand verification is outside the current model scope.

This DINOv3 candidate is a research model and a shadow model. It is not the live DINOv2 production path. `HYPEVAULT_DEPLOYMENT_MODE=production` blocks DINOv3 execution. Research mode allows it. There is no fallback from DINOv2 to DINOv3 or from DINOv3 to DINOv2.

## Out-of-scope use

Do not describe this system as:

- universal luxury-watch authenticity
- arbitrary watch authenticity
- open-set authenticity verification
- a production customer-facing authenticity guarantee

An explicit brand outside the five, a missing brand, or an ambiguous brand returns `UNSUPPORTED_SCOPE` with `decision = null`.

## Dataset

- 30,000 images in the audited corpus
- 15,000 authentic and 15,000 fake
- Five brands
- Corrected split recorded in `ml_rtx5080/experiments/dataset_audit/split_manifest_v2.json`
- Membership hash: `78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678`
- Manifest file SHA-256: `4d0d20c17d12ac29cb299b7b4f369ebcce3c5acd7b2fd5430d61f952b9852eea`
- No verified product or SKU identity
- No marketplace or source metadata
- Product-disjoint evaluation cannot be established

Split counts are train 20,999, calibration 2,998, validation 2,997, and test 3,006.

## Final test

- 3,006 samples
- All five brands
- Calibrated ROC-AUC 1.0
- Calibrated PR-AUC 1.0
- Calibrated F1 1.0
- False-authentic 0
- False-fake 0

This is strong in-distribution evidence and is not evidence of universal luxury-watch authenticity.

## Calibration

- Calibration size: 2,998
- Frozen temperature: `0.24038200410185356`
- Calibrated NLL `2.5221722488311116e-09`, Brier `8.573122380084356e-18`, ECE 0.0 on that split
- Fitting status: fitted
- The temperature was not refit on the final test or on OOD images

## Robustness

- Clean reference false-authentic = 0
- Clean reference false-fake = 0
- 8/9 known stress false-fake events caught by the quality policy
- Resize/recompression remains an uncovered false-fake case (`Label_0_Watches/Richard Mille/9eda705bd8c9.jpg`)
- No false-authentic stress errors were observed in the tested stress suite

## OOD

Phase 35:

- `OOD_FAIL`
- 408 OOD samples
- 337/408 authentic escape under the no-OOD-gate rule
- Unseen luxury escape 0.8814432989690721

Phase 36:

- `WATCH_OOD_UNRESOLVED`
- Independent and underpowered
- 25 holdout images

Phase 37:

- No additional admissible corpus (`WATCH_OOD_V2_CORPUS_READY = false`)

Phase 38:

- `SCOPE_REDUCTION_RECOMMENDED`

`production_ood_threshold = null`

The current system does not have validated open-set protection.
