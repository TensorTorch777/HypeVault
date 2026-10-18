# Freeze and audit protocol

**Outcome of this phase:** no evaluation dataset is frozen.

## Steps, in order

1. Confirm a written authorization names the provider, the records, and the storage location. If it does not, stop. The real cohort stays `NOT_AUDITABLE_UNDER_CURRENT_AUTHORIZATION`.
2. Load only records already approved for the audit. Do not discover extra files. `freeze_audit.audit_synthetic_manifest` rejects a path and rejects image bytes or file paths inside a record.
3. Require Phase 55A eligibility for every primary record: approved license, `ML_EVALUATION`, consent `APPROVED` or `NOT_REQUIRED`, a real retention date, and `deletion_status = RETENTION_ACTIVE`.
4. Fail if one `product_id`, image hash, or duplicate group appears in more than one partition.
5. Check DINOv2 and DINOv3 task-specific training and calibration membership separately. Unchecked membership is `UNKNOWN`, never `PASS`.
6. Treat unknown seller or marketplace identifiers as unresolved. Overlap with a known training or calibration seller or marketplace is a review finding, not a pass.
7. Fail a brand that lacks either independently confirmed authentic products or independently confirmed counterfeit products among passing records.
8. Freeze checkpoint identifiers, preprocessing, aggregation, thresholds, calibration, metric definitions, and acceptance criteria before any score is observed. Those values are not selected in this phase.
9. Record minimum per-brand counts and the false-accept limit only from an approval written before scoring. This protocol does not invent them. `minimum_counts_approved` stays false.
10. Publish aggregate counts and the checksum of those counts. Do not publish private paths or protected item identities.

## Fail-closed results

| Condition | Result |
|---|---|
| Any finding status `FAIL` | Overall `FAIL`. Do not score. |
| Any unresolved membership or seller/marketplace review | Overall `UNKNOWN`. Do not score. |
| Real manifest not authorized | `NOT_AUDITABLE_UNDER_CURRENT_AUTHORIZATION`. Do not open it. |
| Synthetic records pass | Report class `SYNTHETIC_DEMO` and dataset status `NOT_FROZEN`. |

Exclusion reason codes the helper can emit are `PRODUCT_PARTITION_LEAKAGE`, `IMAGE_PARTITION_LEAKAGE`, `DUPLICATE_GROUP_PARTITION_LEAKAGE`, `MODEL_MEMBERSHIP_OVERLAP`, `MODEL_MEMBERSHIP_UNCHECKED`, `SELLER_OR_MARKETPLACE_REVIEW`, `BRAND_MISSING_CLASS`, `RIGHTS_OR_RETENTION`, and `PRIMARY_ELIGIBILITY_FAILED`.

Inference remains unauthorized after every row in this table.
