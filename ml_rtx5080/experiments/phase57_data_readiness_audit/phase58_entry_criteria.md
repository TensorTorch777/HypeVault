# Entry criteria before any later evaluation

**Status:** `METADATA_ONLY — NO DATA ACQUISITION OR INFERENCE AUTHORIZED`

Phase 58, or any later evaluation phase, must not start until the items below are true. This list does not authorize acquisition or inference. `DATA_READINESS` remains `NO_GO_FOR_AUTHENTICITY_CLAIMS`. Independently verified labels remain zero.

No false-accept rate, accuracy bar, or sample-size minimum is set here. Those are approval decisions.

## Documentation and approval

- The evaluation protocol version is frozen and checksummed before any image is received.
- A written approval names the deployment use, the maximum acceptable false-accept risk, the confidence level, the minimum product counts per brand and per class, the abstention policy, and what would support or reject a claim.
- `ACQUISITION_AUTHORIZED` and `INFERENCE_AUTHORIZED` remain false until that approval explicitly changes them. This document does not change them.

## Ground truth and image rights

- Labels used in primary metrics are `AUTHENTIC` or `COUNTERFEIT` from qualified independent reviewers blinded to model outputs.
- Disagreement uses opposite reviewer outcomes and an adjudication outcome that matches `verified_label`.
- Evidence refers to the same physical watch. A detached certificate, listing, auction description, folder name, or model agreement is not enough.
- `license_status` is approved for `ML_EVALUATION`. Consent is `APPROVED` or documented `NOT_REQUIRED`.
- `verified_brand` is not `UNKNOWN`. `retention_until` is a real date. `deletion_status` is `RETENTION_ACTIVE`.
- Schema-valid records exist. Schema validity alone is not entry.

## Manifest and leakage

- A versioned manifest meets `split_manifest_contract.md`, with checksums stored before scoring.
- Every view of one `product_id` is in one partition.
- Exact duplicates and unresolved near-duplicates do not cross partitions.
- Each primary product has `PASS` disjointness from both models' task-specific training and calibration memberships.
- Source overlap is resolved. Known seller and marketplace ids are checked. Unknown ids are not treated as separation.
- Locked final-test and excluded OOD files stay unopened unless a separate registered phase says otherwise. This entry list does not say otherwise.

## Within-brand

- The named brand was present in task-specific training, and the evaluation products of that brand were not.
- Both verified classes exist for that brand, or the claim is explicitly limited to the class that exists and does not state counterfeit performance for the missing class.
- Per-brand counts and rates are reported on physical watches.

## Cross-brand

- Every brand named in the claim has both verified classes.
- Brand is not a proxy for authenticity.
- Per-brand counterfeit results are published beside any aggregate.
- DINOv2 and DINOv3 are compared only on the shared eligible product set. Other products are reported separately.

## Project-task-held-out brand

- The brand is absent from task-specific training, model selection, threshold selection, and calibration.
- The model, preprocessing, aggregation rule, and threshold are frozen before that brand's labels are used.
- The brand is called project-task-held-out, not globally unseen, unless a cited pretraining record shows non-exposure.
- Only brands actually tested are reported.

## Statistics and human review

- Intervals use the physical watch as the unit and state how source clustering is handled.
- The aggregation rule is the one frozen in `metrics_and_acceptance.md`, or a replacement written before evaluation.
- Review/abstain is reported and is not counted as an authentic success.
- Indeterminate labels stay out of binary denominators and are counted.
- No pass/fail sentence is issued until the approval in the first section exists.

If any item above is unmet, the later phase does not run a primary evaluation and does not change production promotion gates.
