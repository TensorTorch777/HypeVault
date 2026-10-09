# Consent and licence record process

## Per agreement

1. Store the signed agreement in the access-controlled evidence store under a `permission_id`.
2. Record:
   - The signatories.
   - The effective date and the expiry or retention end date.
   - The permitted uses, including whether training is allowed.
   - Public-display rights.
   - The withdrawal terms.
   - The name of the legal reviewer who approved the final text.
3. An unsigned or unreviewed draft has `status = NOT_EFFECTIVE`. Images under it are not collected.

## Per image

Every image record references a `permission_id`. The record's `rights` block copies only what the agreement grants. Admission needs all of the following:
- `rights.permission_id` points to an agreement with `status = EFFECTIVE`.
- `rights.research_use` is `GRANTED`.
- `rights.training_use` is `GRANTED` before the image enters a training split.
- The retention end date has not passed.

Any `UNKNOWN` field in `rights` blocks admission.

## Changes

- A withdrawal or expiry sets the agreement to `WITHDRAWN` or `EXPIRED`. Every image under it moves to `disposition = EXCLUDE` with reason `PERMISSION_ENDED` in the next manifest version.
- The manifest is hashed after every change, so the dataset used by any experiment can be reconstructed.
