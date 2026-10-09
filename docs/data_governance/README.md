# Verified data acquisition workflow

Status: `BLOCKED`. Phase 49 admitted 0 independently verified authentic images and 0 verified counterfeit images, from 0 independent source groups, with no confirmed research-use permission. `DATASET_READY_FOR_TRAINING = false`.

These documents describe how a legitimate acquisition would run. They are templates. They are not legal advice, they create no permission, and they do not make any image verified. A responsible party must review and sign the agreements before any image is collected.

| Step | Document |
| --- | --- |
| 1. Contact a qualified examiner, authenticator, or owner group | [outreach_message.md](outreach_message.md) |
| 2. Agree research-use terms (needs legal review) | [contribution_and_research_use_permission_template.md](contribution_and_research_use_permission_template.md) |
| 3. Record consent and licence per contribution | [consent_and_license_record_process.md](consent_and_license_record_process.md) |
| 4. Capture and store images securely | [image_storage_and_retention_policy.md](image_storage_and_retention_policy.md) |
| 5. Verify each item and record the evidence | [adjudication_form.md](adjudication_form.md), [candidate_image_record_v2.schema.json](candidate_image_record_v2.schema.json) |
| 6. Resolve disagreements | [disputed_label_policy.md](disputed_label_policy.md) |
| 7. Admission check | `ml_rtx5080/data_admission.py` (`admission_decision`) |

## Rules that do not change

- A directory name, filename, seller claim, listing label, readable wordmark, file geometry, JPEG table, or model prediction is never label evidence.
- `UNKNOWN` stays `UNKNOWN`. It never becomes a verified value by default.
- The record schema extends `image_label_evidence_v1` (`ml_rtx5080/experiments/dataset_rebuild_v1/image_label_evidence_schema.json`). It adds rights, chain of custody, source groups, file metadata, duplicate status, and an access-controlled verifier reference.
- Before training, a separate protocol must be hashed. It fixes counts, brands, independent source groups, source- and product-disjoint splits, duplicate rules, and acceptance criteria. A pilot that verifies this process is not enough to support broad training.
- The locked five-brand final test is never reused as a holdout for new data.
