# Validation protocol v1

**Status:** `PROTOCOL_ONLY — NO DATA ACQUISITION OR INFERENCE AUTHORIZED`

**Protocol version:** `phase56_validation_protocol_v1`

This document defines three future studies. It does not run them. `DATA_READINESS` remains `NO_GO_FOR_AUTHENTICITY_CLAIMS`. Independently verified labels remain zero.

Frozen identities that a later study would have to re-verify before any forward pass, and must not change in this phase:

- DINOv2 live checkpoint SHA-256 `fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66`, preprocessing `legacy_square_resize_504_imagenet` at 504, native operational policy unchanged.
- DINOv3 experimental checkpoint SHA-256 `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`, preprocessing `resize_pad_square_eval_v1` at 512, frozen temperature `0.24038200410185356`, native shadow policy unchanged.

## Shared rules

**Unit.** The independent unit is a physical watch (`product_id`), not a photograph. Every view, crop, duplicate, and alternate photo of that watch stays in one partition.

**Labels.** Only an independently supported `verified_label` of `AUTHENTIC` or `COUNTERFEIT` can enter a primary metric. `INDETERMINATE` stays out of binary rates and is counted separately. Historical directory labels are retained as historical metadata and are never copied into `verified_label`.

**Evidence.** Reviewers are qualified, independent, and blinded to both models' predictions, logits, and scores, and to each other's decisions until adjudication. Disagreement uses the Phase 55A adjudication rule: opposite reviewer outcomes and an adjudication outcome that matches `verified_label`. Evidence must refer to the same physical watch. Image-use rights and a lawful consent or privacy basis are required. The Phase 55A schema tests do not prove that any real label is correct.

**Leakage.** A primary evaluation product must be absent from both models' task-specific training and calibration memberships, including all of its images. Source, seller, and marketplace overlap is checked as specified in `split_manifest_contract.md`. Unresolved overlap blocks primary eligibility or is excluded from the primary analysis and reported as excluded.

**What is not ground truth.** A public listing, a certificate that is not tied to the examined watch, an auction catalogue description, agreement between the two models, or a folder name.

**Locked data.** This phase does not open, enumerate, hash, or score the locked final-test set or excluded OOD files. A later confirmatory use of the locked final test is a separate registered phase after the protocol, labels, and manifest are frozen and approved.

**Paired comparison.** Where both models can be scored on the same eligible products, that shared cohort is the paired comparison. If eligibility differs because the models' task-specific training memberships differ, the cohorts are reported separately and are not described as a head-to-head result.

## 1. Within-brand validation

**Claim this can support, if later executed and approved.** For a brand that appeared in task-specific training, the model generalizes to other physical watches and sources of that same brand.

**Claim this cannot support.** Generalization to a brand withheld from task-specific training. That is unseen-brand validation.

### Data prerequisites

- Multiple distinct physical watches of that brand.
- Both `AUTHENTIC` and `COUNTERFEIT` verified labels before a within-brand counterfeit-detection claim is made. A brand with only one class is reported as incomplete for that class.
- All views and duplicates of each evaluation product held out of training, development, and calibration.
- Product and source separation recorded against both DINOv2 and DINOv3 task-specific training and calibration memberships.
- Image rights and consent recorded. `deletion_status` is `RETENTION_ACTIVE` and `retention_until` is a real date for any record marked eligible under the Phase 55A schema.

### Permitted calibration data

Development and calibration products of brands other than the evaluation products, already inside the task-specific calibration membership, may be used only under `metrics_and_acceptance.md`. Evaluation labels are not used to choose a threshold, an aggregation rule, or a checkpoint.

### Frozen artifacts

Before scoring: checkpoint hashes, preprocessing identifiers, the product-level aggregation rule, the decision threshold, and this protocol version. They are not revised after evaluation labels are inspected.

### Metrics and uncertainty

Report authentic and counterfeit product counts, source counts, and the metrics in `metrics_and_acceptance.md` for that brand. Confidence intervals use the physical watch as the unit. A review or abstain outcome is not counted as an authentic success.

### Leakage tests

Fail the primary analysis if any evaluation `product_id` or its images appear in either model's task-specific training or calibration records, or if source overlap is unresolved. Near-duplicate links that are unresolved are treated as overlap.

### Failure conditions

- Fewer than the approved minimum of independent products in either class for that brand. The minimum is an approval decision, not a number fixed here.
- Brand identity is a shortcut because one class is supplied by one source or one brand only.
- The aggregation rule or threshold was chosen after seeing evaluation outcomes.

## 2. Cross-brand validation

**Claim this can support, if later executed and approved.** Performance across more than one brand that may already have been seen in task-specific training, with authentic and counterfeit results shown for every brand named in the claim.

**Claim this cannot support.** Unseen-brand generalization. Cross-brand evaluation on brands present in task-specific training does not establish performance on a brand withheld from that training. The model can still use brand-specific appearance learned during task-specific training. A high overall accuracy can hide a brand whose counterfeit products are accepted as authentic. That is why every claimed brand is reported on its own, for both outcome classes, and why an aggregate is never the only result.

### Data prerequisites

- More than one brand.
- For every brand included in a claim: both authentic and counterfeit physical products, independently labeled.
- The evaluation design must not make brand identity a proxy for authenticity. A set in which every authentic watch is one brand and every counterfeit watch is another brand is confounded and cannot support a cross-brand authenticity claim. The Phase 53/54 common held-out cohort is an example of a set that cannot support counterfeit detection: 315 historically authentic-labeled A. Lange & Söhne images and zero historically fake-labeled images, and those directory labels were not independently verified.
- Products and applicable sources held out of both models' task-specific training and calibration memberships.

### Reporting

- Per-brand product counts, source counts, both class counts, and the metric set in `metrics_and_acceptance.md`.
- An overall aggregate in addition to, not instead of, the per-brand results.
- If a brand lacks a verified counterfeit or authentic product, that brand is reported as incomplete. It is not filled with historical folder labels.
- Paired DINOv2 versus DINOv3 numbers only on the shared eligible cohort. Otherwise the cohorts are separate and are not called a direct comparison.

### Permitted calibration data, frozen artifacts, uncertainty, leakage, and failure

Same shared rules as within-brand validation. An additional failure condition: publishing an aggregate that is not accompanied by the per-brand counterfeit result for every brand named in the claim.

## 3. Unseen-brand validation

**Name.** Call the withheld brand a **project-task-held-out brand**. DINOv2 and DINOv3 foundation pretraining may have included images of that brand. That exposure is unknown unless a cited record shows it. Do not call the brand globally unseen to the pretrained model without that evidence.

**Claim this can support, if later executed and approved.** Generalization to a brand withheld from task-specific training, task-specific model selection, threshold selection, and calibration.

**Claim this cannot support.** A statement that the pretrained backbone never observed the brand, unless pretraining membership is actually known.

### Leave-one-brand-out rule

For each brand with sufficient independent evidence:

1. Exclude every product and image of that brand from task-specific training, model selection, threshold selection, and calibration.
2. Freeze the model weights, preprocessing, product-level aggregation rule, and decision threshold before any held-out brand label is used.
3. Score only that held-out brand.
4. Repeat for the next brand with sufficient evidence. Do not select the brand after seeing which one looks best.
5. Report evidence only for brands actually tested. Do not impute results for brands that were not held out.

### Data prerequisites

Both verified classes for the held-out brand, unless the report explicitly limits the claim to the class that exists. Product and source separation still apply. Brands without enough independently verified products are listed as not tested.

### Permitted calibration data

Only products from brands other than the held-out brand, and only inside the approved development or calibration partition. The held-out brand's labels are not used to pick a threshold or a checkpoint.

### Frozen artifacts, metrics, uncertainty, leakage, and failure

Same shared rules. Additional failure conditions:

- The held-out brand appears in task-specific training, selection, or calibration.
- The threshold or aggregation rule changes after the held-out evaluation is seen.
- The report describes a task-trained brand as project-task-held-out.
- The report calls the brand globally unseen without evidence about foundation pretraining.

## Why the three protocols stay separate

| Question | Protocol that can answer it |
| --- | --- |
| New watches of a brand the task-specific model trained on | Within-brand validation |
| Several such brands, without hiding a weak brand inside an average | Cross-brand validation |
| A brand withheld from task-specific training, selection, and calibration | Unseen-brand validation |

Cross-brand evaluation on known brands does not answer the third question. The task-specific head and any task-specific updates can learn that brand's appearance. Holding out new products of a known brand tests new products, not a new brand.
