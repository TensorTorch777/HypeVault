# Verified authenticity data pilot

Phase 49 asked whether a small, legally usable set of independently verified authentic and counterfeit watch images can be built. The protocol was hashed before any admission decision. No image was downloaded, and no model was trained or scored.

Protocol SHA-256 `42d03f08827a7657028e1cf0d44973ec779f79e7dd4a271b10deb80aa48f7112`.

Manifest SHA-256 `bfb27c410b5c608fec8a0ba1d452a67d33ce26f459795ef33168d6607b3a0c92`.

## What was examined

Sixteen sources were written into the candidate manifest and rejected. Image fields that were not obtained are `UNKNOWN`. Rejection is not a verified label.

The existing five-brand catalog fails the rule. Phase 48 already showed that its labels are directory names and that counterfeit files share one 512-pixel JPEG export. That catalog was not reused.

Public watch sets that were checked again fail for a specific missing piece:

- Chrono-analyzer has an Apache 2.0 card and brand folders. It does not label counterfeits.
- Roboflow watch exports have a CC BY 4.0 claim. They are detection or derivative sets, and the Rolex exports were already used in Phase 35.
- TagMyWatch and the Hugging Face listing tables come from marketplaces. A listing is not an authenticity examination, and the photograph rights are not established.
- The Hugging Face product-authenticity set is synthetic text, including model-written verdicts.
- TrustWatch describes a private genuine-versus-counterfeit collection and scores images with its own model. No dataset, license, or verifier record is available to this project. Another model's score is not ground truth.
- GoldNet, FAMOS, and the IEEE QR set have authenticity-style labels for gold, package microstructures, and printed codes. They are not watch photographs.

No partnership with a watchmaker, authenticator, or dealer is recorded. No owner has granted permission to photograph a watch for this pilot. No counterfeit example has an institutional record in the repository.

## Integrity

No pilot file was hashed, so exact and perceptual duplicate checks against the catalog and against Phase 35 and Phase 36 were not run. The locked final-test images were not opened. The shared original-plus-resize-pad policy stays the rule for a future acquisition. It was not applied.

## Decision

`NO_GO`

The target of 30 authentic and 30 counterfeit examples was not met. The decision does not rest on that count. No source supplied both a research-use right and an independent authenticity method, and no named party has a concrete data commitment outstanding.

`DATA_PILOT_STATUS = NO_GO`

`AUTHENTIC_VERIFIED_COUNT = 0`

`COUNTERFEIT_VERIFIED_COUNT = 0`

`INDEPENDENT_SOURCE_GROUPS = 0`

`LICENSE_AND_PERMISSION_CONFIRMED = false`

`LABEL_PROVENANCE_COMPLETE = false`

`DATASET_READY_FOR_TRAINING = false`

`MODEL_CHANGED = false`

`CALIBRATION_CHANGED = false`

`FINAL_TEST_TOUCHED = false`

`PRODUCTION_PROMOTION_ALLOWED = false`

## Recommendation

Model development should not resume on the current catalog or on any of the rejected public sets.

The practical route that remains is a written arrangement with a qualified examiner, or owner-permission photography paired with that examination. Each admitted watch needs a source, a product or listing id, an image right, and a verification record that does not come from the model. Counterfeit examples need affirmative evidence of the same kind. Until those records exist, there is no training set to rebuild.
