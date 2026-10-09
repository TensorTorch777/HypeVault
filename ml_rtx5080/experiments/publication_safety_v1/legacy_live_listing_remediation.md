# Legacy live listing remediation

Existing live listings are historical product state and are not automatically reclassified as authentic or fake by this phase.

No listing was deleted, rejected, or moved to pending.

## LOW — 20 listings

Evidence: `verdict = AUTHENTIC`, confidence exactly `0.965`, no stored image, and one shared `created_at` of `2026-04-23T04:45:09.378287+00:00`. That is the historical seed writer. The current seed script no longer sets `live` or `AUTHENTIC`. Ten of the names are watches that are still in the seed list, and ten are sneakers from an older seed batch in the same insert.

Recommended action: Option A. Leave them live and do not present the stored verdict as marketplace authentication.

Urgency: low for catalog removal. The displayed claim should stay separate from `status = live`.

Automated status change: not justified.

## HIGH — 19 listings

Evidence: each has a stored image and a confidence value other than the seed constant `0.965` or the import constant `0.94`. The legacy `POST /verify/authenticate` path is the only application code that stored an image together with a model verdict, and before Phase 42 a non-`FAKE` verdict set `status = live`. This is process risk, not a finding that any watch is fraudulent.

Recommended action: Option B for a later explicit decision. They can be labeled legacy-unverified in the interface. This phase does not change their status.

Urgency: the product page previously rendered `Live · Authentic` from `verdict = AUTHENTIC`. That label is now `Legacy screening`. Unpublishing is not justified by an existing business or legal rule.

Automated status change: not justified.

## MEDIUM — 0 live listings

No live row had a verdict whose publication source was otherwise unclear.

## IMPORT and manual publication — 0

The Chrono24 import signature is absent from live rows. `PATCH /listings/{id}` cannot set `status` or `verdict`. No admin publish tool was found.

## Option D — remove from catalog

Not recommended. No existing policy requires removal.
