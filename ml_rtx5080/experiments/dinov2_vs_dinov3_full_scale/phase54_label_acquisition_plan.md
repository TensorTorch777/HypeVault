# Phase 54 label-acquisition plan

This plan does not authorize inference. Phase 53 and the Phase 54 manifest audit both leave `DATA_READINESS = NO_GO_FOR_AUTHENTICITY_CLAIMS`. The shared held-out cohort has no fake-labeled image, no independently verified label, and no known physical-product id.

## 1. Independent verification rubric

A label may enter the primary benchmark only after evidence is recorded. The model under test, the other frozen model, the directory name, visual impression alone, and agreement between two AI systems are not evidence.

| Outcome | Required evidence |
| --- | --- |
| `AUTHENTIC` | A documented provenance chain for that physical watch, or a written examination by a qualified specialist. The record names the evidence category, not a person's identity. |
| `COUNTERFEIT` | A qualified independent examination, or an equivalent documented finding that states what was examined and why it is counterfeit. |
| `INDETERMINATE` | Anything uncertain, incomplete, or disputed. These rows are excluded from primary binary metrics. |

Directory names stay `HISTORICAL_DIRECTORY_LABEL`. They are not copied into `verified_label`.

## 2. Two-reviewer process

1. Two reviewers qualified for watch authentication work independently.
2. Each reviewer is blinded to both models' logits, scores, and decisions, and to the other reviewer's decision.
3. Each record stores an evidence category and an outcome. It does not store the reviewer's name, account, or contact details. Use a role code such as `reviewer_a` / `reviewer_b` that is not a personal identifier.
4. Agreement on `AUTHENTIC` or `COUNTERFEIT` can stand.
5. Any disagreement, including disagreement with `INDETERMINATE`, goes to a pre-assigned adjudicator who sees the two evidence records and not the model outputs.
6. The adjudicator's outcome is the label. If the adjudicator cannot decide, the label remains `INDETERMINATE` and leaves the primary metric set.

## 3. Sampling

The sampling unit is a physical watch, not a file. Every supported brand needs both authentic and counterfeit products before a per-brand claim is made. Supported brands in the current corpus are A. Lange & Söhne, Audemars Piguet, Patek Philippe, Richard Mille, and Vacheron Constantin.

All views, crops, reposts, near-duplicates, and other images of the same physical watch share one product id and one split. A hash group from `split_manifest_v2.json` is not that product id. Product ids are assigned from the verification record, or stay `UNKNOWN`.

Images are drawn from the verification pool before any model score from this benchmark is computed. Phase 53 predictions are not a sampling frame.

## 4. Source and product separation

Where the law and the consent record allow it, store a pseudonymous seller, marketplace, and source id. Do not store account names or serial numbers in the committed report.

Evaluation products and sources must be absent from both frozen models' training and calibration memberships. Because those checkpoints are already trained, there is no retraining step that can repair overlap. A brand that appeared in training is not an unseen brand. Brand-held-out numbers for a brand the model trained on must not be described as unseen-brand generalization.

The current manifests do not support this separation: product, seller, source, and marketplace are `UNKNOWN`, and the live DINOv2 validation folder is a brand directory the experimental model also used.

## 5. Sample-size policy

These targets are fixed before the future inference. They are not revised after seeing those predictions. Phase 53 cannot supply a false-authentic rate, because its paired cohort has no fake-labeled image.

- Primary safety statement: a one-sided 95% upper bound on the false-authentic rate among independent counterfeit products.
- With zero observed false-authentic errors in `n` independent counterfeit products, that bound is approximately `3/n`. An overall bound near 1% needs about 300 counterfeit products. The same bound for one brand needs about 300 counterfeit products of that brand.
- Use a Clopper-Pearson interval on the product-level rate, not the image-level rate. If several images share a product, the interval is on the product outcome. Do not treat those images as independent.
- Paired comparison: McNemar on the pre-declared binary-decision subset, two-sided alpha 0.05, target power 0.80. The pre-registered discordant-pair assumption is that 10% of products are discordant and that a meaningful split of those discordant products is 70/30. This assumption was not estimated from Phase 53. If the realized discordant count is below the count required by that assumption, the paired test is reported as underpowered and cannot name a winner.
- Minimum subgroup: no per-brand claim unless that brand has both verified classes and meets the product count for the claim being made.

This is not a claim that 300 products are representative of the market.

## 6. Governance

Before images are added to the verification pool:

- Record the license or permission category that allows evaluation use.
- Get informed permission where the source requires it.
- Keep the committed artifacts free of serial numbers, account identifiers, and reviewer identities.
- Store evidence notes in the local ignored directory, with access limited to the reviewers and the adjudicator.
- Define retention and deletion before acquisition. Verification records that are no longer needed for the frozen protocol are deleted rather than copied into git.
- The evidence trail is the rubric outcome, evidence category, role codes, and adjudication outcome. It is not a dump of private correspondence.

## 7. Go / no-go

Primary confirmatory inference stays blocked until all of the following are true:

1. Both binary classes have independently verified labels under the rubric above.
2. Eligible products are absent from both models' training and calibration memberships.
3. Product and source identities are known, and images of one product cannot cross splits.
4. The pre-registered product counts for the claim are met.
5. The benchmark protocol, sample manifest, and analysis plan are frozen.
6. The locked final test remains unopened until a separate one-time phase is explicitly approved.

If any item fails, keep `DATA_READINESS = NO_GO_FOR_AUTHENTICITY_CLAIMS`.

## Pre-registered benchmark design (not executed)

When, and only when, the go criteria pass:

- Score both frozen checkpoints on the same eligible images. DINOv2 uses `legacy_square_resize_504_imagenet` at 504. DINOv3 uses `resize_pad_square_eval_v1` at 512. Each model keeps its native decision policy, including DINOv3 `REVIEW`.
- Before the first forward pass, require DINOv2 SHA-256 `fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66` and DINOv3 SHA-256 `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`. A mismatch stops the run.
- Report confusion counts, accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, and PR-AUC from the raw logit in the fake-positive direction. Report false-authentic rate and false-fake rate with the verified-product denominator written next to the numerator. Report DINOv3 review count, review rate, decision coverage, and selective accuracy. Report paired agreements, disagreements, and error slices.
- `REVIEW` is its own outcome. It is not relabeled authentic or fake, and it stays in the coverage denominator.
- Compare paired binary decisions with McNemar on the subset where both models returned `AUTHENTIC` or `FAKE`. Use product-level cluster bootstrap or another clustered interval when a product contributes more than one image. Images of one watch are not independent observations.
- Do not compare the two raw logits, or the DINOv2 sigmoid and the DINOv3 temperature-scaled sigmoid, as one calibrated scale.
- Leave the locked final test sealed. Any use of it is a separate registered phase after this protocol, the labels, the sample manifest, and the analysis plan are frozen and approved.
