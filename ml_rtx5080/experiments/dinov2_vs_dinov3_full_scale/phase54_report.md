# Phase 54 data-readiness audit

DATA_READINESS = NO_GO_FOR_AUTHENTICITY_CLAIMS

No confirmatory model comparison was run. Phase 53 commits `888d97ebf0e9a41d1cadc873d2ed34f1254dc4d0` and `22d9384bb640f1537ceed9d499f3f6dea1f4987e` were checked and not rewritten. The shared held-out cohort is 315 images, 315 authentic-labeled and 0 fake-labeled, all from A. Lange & Söhne. False-authentic rate, ROC-AUC, and PR-AUC are undefined. Independently verified labels: 0.

## Membership

DINOv2 `fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66` is tied to `ml_rtx5080/checkpoints/split_manifest.json` by the Triton identity record and by the training config's seed and validation fraction. That manifest has no calibration split (`NOT_RECORDED`) and includes 10 sneaker folders outside the watch corpus. DINOv3 `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f` is tied to `split_manifest_v2.json` by the training config's membership hash and by the calibration file that names `epoch_018.pt`.

The watch manifest contains 30000 samples, of which 3006 are the locked final test. Those image files were not opened. The primary count matches Phase 53: true.

DINOv2 rows by DINOv3 columns:

| DINOv2 \ DINOv3 | calibration | test | train | validation |
| --- | ---: | ---: | ---: | ---: |
| train | 2697 | 2723 | 18898 | 2682 |
| validation | 301 | 283 | 2101 | 315 |

The validation/validation cell is the primary cohort. The validation/test cell stays inside the locked split and was not opened. The train/validation cell is DINOv3 validation that the live DINOv2 checkpoint already used for training.

## Contradiction that was not smoothed over

`docs/ML_PRODUCTION_AUDIT.md` says validation is Vacheron Constantin authentic-labeled and Richard Mille fake-labeled. That description matches `checkpoints_watches/split_manifest.json`. The checkpoint beside that manifest hashes to `19e832cea9295975bf798e0a0d372a3a5a20d3f12c949ad32eb82949afb8cf13`, which is not the live DINOv2 checkpoint. Phase 53's Lange-only intersection stands. The older VC/RM statement does not describe the live model.

## Provenance

Product, seller, source, and marketplace identifiers are `UNKNOWN`. Verification, evidence, adjudication, license, and consent are `NOT_RECORDED`. Duplicate-group ids in the manifest are hash groups, not physical watches. Directory labels are not ground truth.

Exact duplicate groups among non-test images with a manifest SHA-256: 2.

## Readiness reasons

- primary_cohort_missing_a_binary_class
- no_independently_verified_labels
- product_id_unknown
- source_or_product_disjointness_not_established
- independent_product_count_below_preregistered_minimum

The future paired benchmark in `phase54_label_acquisition_plan.md` is not authorized to start.
