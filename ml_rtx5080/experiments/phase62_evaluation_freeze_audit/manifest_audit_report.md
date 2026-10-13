# Manifest audit report

**Report class:** `SYNTHETIC_DEMO`

**Dataset status:** `NOT_FROZEN`

**Real cohort:** `NOT_AUDITABLE_UNDER_CURRENT_AUTHORIZATION`

## What was checked

An in-memory pair of synthetic records, one labeled `AUTHENTIC` and one labeled `COUNTERFEIT`, with empty checked membership views for DINOv2 and DINOv3 training and calibration. The helper returned overall `PASS`, record count 2, eligibility counts PASS 2 / FAIL 0 / UNKNOWN 0 / NOT_ELIGIBLE 0, and finding count 0.

The aggregate checksum, computed without the checksum field itself, is `8861c3c6b574138ad458eb7997b2628458e9251a5abbecb8476ff163d8e9af27`. The same object is stored in `manifest_checksums.json`.

## What was not checked

No real product, source, seller, marketplace, or model-membership list was read. Locked final-test and excluded out-of-distribution membership remain unauditable. The 315 historically authentic-labeled A. Lange & Söhne images are not a frozen evaluation set. Independently verified labels remain zero.

Minimum counts and acceptance criteria are not approved. `acquisition_authorized` and `inference_authorized` are false. `DATA_READINESS` remains `NO_GO_FOR_AUTHENTICITY_CLAIMS`.

Separate unit tests cover unchecked membership, a brand missing a class, and refusal of a dataset path. Those cases are not the checksummed demo pair.
