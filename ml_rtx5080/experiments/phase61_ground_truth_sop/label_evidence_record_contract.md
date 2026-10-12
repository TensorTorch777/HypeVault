# Label and evidence record contract

A future evidence record connects one physical watch to its views without placing sensitive source material in Git.

## In the approved store, not in Git

- Raw serial numbers and customer identities.
- Reviewer legal names, if they are personal data.
- Original certificates, private reports, and images.
- Correspondence with a provider.

## Fields that may later be committed only as pseudonymous aggregates or a schema-valid ledger outside the image store

These names match `evidence_ledger.schema.json`:

- `sample_id`, `image_sha256`, `product_id`, `source_id`
- `verified_label` with values `AUTHENTIC`, `COUNTERFEIT`, or `INDETERMINATE`
- `reviewer_a_outcome` and `reviewer_b_outcome`
- `evidence_category_a` and `evidence_category_b`
- `adjudication_status` and `adjudication_outcome`
- `license_status`, `consent_status`, `permitted_purposes`
- `retention_until`, `deletion_status`
- `source_product_disjointness`
- `eligible_primary_benchmark`

`historical_directory_label` may be retained as metadata. It must not be copied into `verified_label`.

## Linkage rule

Every image, crop, and duplicate view of one physical watch uses one `product_id`. Unknown product identity is not primary-eligible. Seller and marketplace identifiers, when lawfully known, are recorded. `UNKNOWN` is not evidence that sources are separate.

## Sample size

No minimum count is adopted here. The Phase 56 illustration of roughly 300 counterfeit products and a `3/n` bound is not a target and not a guarantee. A later statistical plan has to be approved before anyone claims a sample is sufficient.

## Current count

Independently verified labels: 0. This contract does not increment that count.
