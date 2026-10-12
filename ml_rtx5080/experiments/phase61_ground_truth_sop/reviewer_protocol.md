# Reviewer protocol

**Status:** procedure only. No reviewer has been appointed by this document.

## Competency and conflicts

A reviewer must have a documented scope for the brands and references they examine. The record of qualifications stays in the approved store. A reviewer who is the seller, the marketplace operator, or otherwise interested in the outcome of a specific watch declares that conflict and does not review that watch.

Blinding includes model predictions and the other reviewer's outcome. It also includes any HypeVault score or threshold.

## Allowed outcomes

Each reviewer records one of:

- `AUTHENTIC`
- `COUNTERFEIT`
- `INDETERMINATE`

`NOT_REVIEWED` means the review did not happen. It is not a label.

## Evidence categories

Each reviewer names one category from the Phase 55A schema:

- `MANUFACTURER_PROVENANCE`
- `AUTHORIZED_SERVICE_RECORD`
- `QUALIFIED_PHYSICAL_EXAMINATION`
- `DOCUMENTED_CHAIN_OF_PROVENANCE`
- `OTHER_DOCUMENTED`
- `NOT_RECORDED`

`NOT_RECORDED` cannot support a primary-eligible record.

## Adjudication rules

1. Two reviewers examine independently.
2. If both choose the same binary outcome and the evidence is accepted, `adjudication_status` is `AGREED` and `verified_label` matches that outcome.
3. If one chooses `AUTHENTIC` and the other `COUNTERFEIT`, a third qualified person may adjudicate. `adjudication_status` is `ADJUDICATED` only when the adjudication outcome equals `verified_label` and the two reviewer outcomes are opposites.
4. If either reviewer chooses `INDETERMINATE`, the watch is not given a binary verified label for the primary benchmark.
5. Disagreement is not resolved by a model score.
