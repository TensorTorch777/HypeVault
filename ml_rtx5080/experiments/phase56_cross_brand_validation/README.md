# Phase 56 — Cross-brand validation protocol

**Status:** `PROTOCOL_ONLY — NO DATA ACQUISITION OR INFERENCE AUTHORIZED`

Writing this protocol does not mean either frozen model has passed within-brand validation, cross-brand validation, or unseen-brand validation. No authenticity claim is supported by this directory.

`DATA_READINESS` remains `NO_GO_FOR_AUTHENTICITY_CLAIMS`. `ACQUISITION_AUTHORIZED` is false. `INFERENCE_AUTHORIZED` is false. Independently verified labels remain zero.

## Purpose

Define how a future, separately authorized study would show three different claims:

1. **Within-brand validation.** Generalization to new physical watches and sources for brands that were represented in task-specific training.
2. **Cross-brand validation.** Performance across multiple brands, with authentic and counterfeit outcomes reported separately for every brand on which a claim is made.
3. **Unseen-brand validation.** Generalization to a brand withheld from task-specific training, model selection, and calibration.

These are not interchangeable. A result on brands the task-specific model already trained on is not unseen-brand generalization.

## Layout

| File | Role |
| --- | --- |
| `validation_protocol_v1.md` | The three protocols, their data rules, and their failure conditions. |
| `split_manifest_contract.md` | The future manifest that would make product, source, and split membership auditable. |
| `metrics_and_acceptance.md` | Metrics, uncertainty, and the decisions that still require approval. |

## Prerequisites that are not met

Phase 53/54 and the merged Phase 55A preflight established the following facts. This protocol does not change them.

- Historical folder labels are not independently verified ground truth. A public listing, a certificate detached from its physical watch, an auction description, a model prediction, or a directory name is not authenticity.
- Independently verified labels remain zero until qualified evidence is collected under a later written authorization.
- The audited common held-out cohort had 315 historically authentic-labeled A. Lange & Söhne images and zero historically fake-labeled images. It cannot substantiate counterfeit detection.
- Source/product leakage and brand/authenticity confounding must be measured before any claim.
- Phase 55A's evidence-ledger schema is on `main`. Passing its tests shows only that the metadata contract is enforced. It does not show that any watch label is correct or that either model generalizes.

Future ground truth, when a later phase is authorized, must have documented image-use rights, a lawful consent or privacy basis, qualified independent reviewers, blinded assessment, evidence tied to the same physical watch, adjudication when reviewers disagree, and auditable source/product disjointness from both frozen models. Eligible primary-benchmark records must also satisfy the fail-closed Phase 55A schema, including a verified brand, a real retention date, and `deletion_status` of `RETENTION_ACTIVE`.

## Completion conditions for this phase

This phase is complete only as documentation when:

- The three validation types are defined separately and are not treated as the same claim.
- Foundation-model pretraining exposure is distinguished from task-specific training exposure.
- No numeric acceptance bar is presented as an established project fact.
- No dataset directory is read, no locked final-test or excluded OOD file is opened, and no model is trained, calibrated, or run.

This phase is not complete as a validation study. No cohort has been scored.
