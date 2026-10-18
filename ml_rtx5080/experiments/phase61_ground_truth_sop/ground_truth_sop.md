# Ground-truth standard operating procedure

**Status:** procedure only. Do not execute the collection steps until a written approval names the provider, the records, the storage location, and the people allowed to handle them.

## Chain of custody

1. Assign a pseudonymous `product_id` before any image is stored. Do not put the raw serial, customer name, or original report into Git.
2. Record who received the physical watch or the evidence bundle, when, and where it is stored. That log stays in the approved access-controlled store.
3. Photograph or receive only images covered by the written rights grant. Multiple views of one watch share the same `product_id`.
4. Compute `image_sha256` in the approved store. Commit only aggregate counts and checksums of a future manifest, not the images.
5. If custody is unclear, mark the record ineligible. Do not reconstruct custody from a listing or a folder name.

## Rights verification

Before a record can be primary-eligible, the schema fields must already say:

- `license_status = APPROVED`
- `consent_status` is `APPROVED` or `NOT_REQUIRED`, with the reason stored outside Git
- `permitted_purposes` contains `ML_EVALUATION`
- `retention_until` is a real date
- `deletion_status = RETENTION_ACTIVE`

A missing grant is `PENDING` or `NOT_RECORDED`, and `eligible_primary_benchmark` stays false.

## Reviewer blinding

Reviewers do not see model scores, the other reviewer's outcome, or the historical directory label while they examine the evidence. The historical label may be stored as metadata and must not be copied into `verified_label`.

## Evidence linkage

The written evidence must refer to the same physical watch as the images. An online listing, a detached certificate, a directory name, or agreement between models is not linkage.

## Adjudication

Follow the merged Phase 55A schema:

- `AGREED` requires both reviewers to match each other and to match `verified_label`, which is `AUTHENTIC` or `COUNTERFEIT`.
- `ADJUDICATED` requires the reviewers to have opposite outcomes and the adjudication outcome to equal `verified_label`.
- `INDETERMINATE` is a valid reviewer outcome and is not forced into a binary class. It is not primary-eligible as a counterfeit or authentic label.

## Data security

Raw serials, customer information, reviewer identities, original reports, and images stay in the approved store. Git receives the procedure, schema, and later safe aggregates only.

## Escalation

Stop and return the record to the custodian if rights are denied, a reviewer declares a conflict, the watch identity is ambiguous, or anyone proposes opening locked final-test or excluded out-of-distribution files. Those files are outside this procedure.
