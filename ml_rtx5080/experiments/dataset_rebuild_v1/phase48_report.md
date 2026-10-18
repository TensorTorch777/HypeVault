# Dataset rebuild protocol

Phase 48 defines how a later dataset can be admitted. It does not train a model, assign a split, or change a directory label. The protocol was hashed before the visual sample and the candidate manifest were written.

Protocol SHA-256 `0c04b6e182db916936142c3fd539148641778e0d10041b40cdb3def7c49473bc`.

## Class geometry

On the 26,994 non-test images, a rule fixed in the protocol before the census — counterfeit if and only if the file is 512×512 — agrees with the directory label on 26,993 files. All 13,500 counterfeit files are 512×512. One authentic file is 512×512, and it is a progressive JPEG with a different quantization table.

Every counterfeit file is a baseline JPEG using one shared quantization table, matching the standard IJG quality-95 tables exactly. Authentic files use 623 other JPEG tables plus 240 files with no JPEG table (211 WebP, 28 PNG, 1 TIFF). The counterfeit table also appears on 127 authentic files at other sizes, so the table alone does not separate the folders. Progressive JPEG appears on 5,663 authentic files and on no counterfeit file. File-size medians remain lower for the counterfeit folder in every brand, which follows from the shared 512-pixel export.

The 30-image visual sample, seed 48 and three images per brand and label, was drawn after the hash. Twenty-nine images were viewed. One authentic WebP file could not be decoded by the viewer; its measurements still come from Pillow. Viewed counterfeit images fill the square. No letterbox or flat padding bar was seen. Several authentic images use a white or black studio field or a feathered cutout. Sample median Laplacian variance is 1046.8 for the authentic cell and 2479.3 for the counterfeit cell. Those two medians belong to this sample of 30.

No classifier was fit. The research loader still resizes every image to 512 before the network, so raw width is not a numeric input. The census shows a class-wide export convention. It does not show that the frozen network uses that convention.

## Labels

The catalog manifest has no source, product, or verifier field. All 30,000 rows therefore fail independent verification:

| Provenance | Rows |
| --- | ---: |
| INDEPENDENTLY_VERIFIED | 0 |
| UNVERIFIED | 29,942 |
| REQUIRES_EXPERT_REVIEW | 58 |

Directory names, wordmarks, and model scores were not treated as proof, and no directory label was changed.

## Quarantine

Source files were left in place. Historical evaluation artifacts were left in place.

| Quarantine reason | Files |
| --- | ---: |
| JS monogram copied across authentic brand folders | 9 |
| Expert review, including 31 Phase 47 flags and 18 Phase 48 disputes | 49 |

The other non-test rows stay `RETAIN_UNVERIFIED`. The 3,006 locked test rows are `SEALED_FINAL_TEST` and were not opened. Every row has `training_eligible` false and `admitted_label` null. No new split was assigned.

## Verification and the future dataset

An admitted label needs a source identifier, a product or listing identifier, a verifier independent of folder names and of any model, a method, and an evidence identifier. If those are missing, the image stays `UNVERIFIED`.

The shared geometry policy is `shared_original_plus_resize_pad_square_v1`: store acquired files at their original geometry, and apply one loader transform to every class and brand — long side 512, bicubic, centered ImageNet-mean padding. Exact duplicates stay in one source group. Near-duplicates without a listing id stay unverified. Collages, component-only images, placeholders, and ambiguous brand identity stay in expert review.

`LABEL_VERIFICATION_PROCESS_DEFINED = true`

`CLASS_GEOMETRY_CONFOUND_DOCUMENTED = true`

`UNVERIFIED_IMAGES_QUARANTINED_FROM_FUTURE_TRAINING = true`

`DATASET_READY_FOR_TRAINING = false`

`MODEL_CHANGED = false`

`CALIBRATION_CHANGED = false`

`FINAL_TEST_TOUCHED = false`

`PRODUCTION_PROMOTION_ALLOWED = false`

## Evidence still required

A training rebuild can start only after each admitted image has the schema fields above, both classes are stored without a class-specific resize, and listing ids exist so related images can be kept together. Component images, placeholders, and the current expert-review set need a recorded decision before they can be admitted. The locked final test stays sealed until that admission protocol is hashed. Until then the candidate manifest has no training rows.
