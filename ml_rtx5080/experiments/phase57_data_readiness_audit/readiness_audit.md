# Phase 57 readiness audit

**Status:** `METADATA_ONLY — NO DATA ACQUISITION OR INFERENCE AUTHORIZED`

`DATA_READINESS` remains `NO_GO_FOR_AUTHENTICITY_CLAIMS`. Independently verified labels remain zero. Facts below are taken from the files named in `README.md`. Interpretation is marked as such.

## What is documented

Phase 56 states three separate protocols: within-brand, cross-brand, and project-task-held-out brand. It also states that cross-brand results on brands used in task-specific training are not unseen-brand generalization, and that foundation-pretraining exposure is unknown unless a record shows it. Source: `validation_protocol_v1.md`.

Phase 56 states product-level metrics, a default aggregation rule that must be frozen before evaluation, and a list of acceptance decisions that are not yet made. Source: `metrics_and_acceptance.md`.

Phase 55A states a fail-closed schema: an eligible record needs a verified brand, a real retention date, `deletion_status` of `RETENTION_ACTIVE`, approved rights, evidence categories, a resolved label, and `PASS` disjointness for both models. An adjudicated record needs opposite reviewer outcomes and a matching label. Source: `evidence_ledger.schema.json`. Passing tests of that schema is not evidence that any watch is eligible.

Phase 55A's tracker lists four leads, all `NOT_SENT`, none approved. WatchCSA is `HOLD_DO_NOT_UPLOAD`. Rolex Certified Pre-Owned is described as positive-only. Phillips catalogue text is not primary evidence. Source: `source_tracker.csv` and `source_due_diligence.md`.

The live DINOv2 checkpoint hash `fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66` is in `infra/triton/dinov2_vitb14_live/identity.json`. Phase 54 ties that hash to a folder split with no calibration split, and ties DINOv3 `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f` to `split_manifest_v2`. This audit did not reopen the image membership file.

## Ground truth that remains unmet

Phase 54 records `independently_verified` as 0, `directory_labels_are_ground_truth` as false, and product, source, seller, marketplace, license, and consent as `UNKNOWN` or `NOT_RECORDED`. The shared held-out cohort is 315 historically authentic-labeled A. Lange & Söhne images and zero historically fake-labeled images. False-authentic rate, ROC-AUC, and PR-AUC are null. Those are measurements of historical directory labels, not verified authenticity, and they do not support counterfeit detection.

No document inspected here shows both authentic and counterfeit verified examples for any brand.

## Separation

Physical-product separation is not demonstrable. Phase 54 says product ids are unknown and that duplicate groups are hash groups, not watches.

Source, seller, and marketplace separation is not demonstrable. Those fields are `UNKNOWN`. Unknown is not proof that a source was held out.

Task-specific overlap is only partly described. Phase 54's published table says 2,682 images in DINOv3's validation split sit in DINOv2 training folders, and DINOv2 has no calibration split on the live checkpoint. That is a documented summary of historical image splits. It is not a product-level `PASS` for both models. Checking the locked 3,006-row membership again would require opening a protected list, so that recheck is `NOT_AUDITABLE_UNDER_CURRENT_AUTHORIZATION`.

## Rights, review, and retention

The schema and the Phase 56 contract say what rights, consent, retention, deletion, review, and adjudication must look like. The tracker and Phase 54 metrics say those facts are not present for any acquired item. No reviewer qualification file, consent record, or retention date for a real watch was found in the authorized inputs.

## By validation type

**Within-brand.** The definition is documented. Empirical requirements are unmet: no verified both-class set, no product ids, no source ids, no rights on real items, and no per-brand counterfeit result. Membership overlap is only summarized at folder/split level.

**Cross-brand.** The definition is documented, including the ban on hiding a weak brand inside an aggregate and the ban on calling known-brand results unseen-brand generalization. Empirically, verified multi-brand both-class coverage is missing. The only common held-out cell in Phase 54 is one historical class and one brand, which the Phase 56 protocol would treat as unable to support the claim. No lead is approved to supply the missing classes.

**Unseen-brand (project-task-held-out).** The leave-one-brand-out rule is documented. No verified brand has been shown to be withheld from both models' task-specific training, selection, and calibration. Phase 54 explicitly says a Vacheron Constantin / Richard Mille folder split belongs to checkpoint `19e832ce`, not the live DINOv2 checkpoint, so that older split is not unseen-brand evidence for the live model. Foundation-pretraining exposure was not audited; the protocol forbids calling a brand globally unseen without that evidence, and this audit did not open pretraining corpora.

## Protected data

This audit did not open image directories, `local_results/`, training payloads, or manifests that list locked final-test or excluded OOD paths. Where a fact needed that access, the matrix says `NOT_AUDITABLE_UNDER_CURRENT_AUTHORIZATION` instead of guessing.
