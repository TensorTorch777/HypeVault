# Phase 59 — Synthetic manifest validator

**Status:** `VALIDATOR_ONLY — NO DATA ACQUISITION OR INFERENCE AUTHORIZED`

This directory checks future evaluation-manifest metadata that is already in memory. It does not read watch images, dataset directories, `local_results/`, or locked final-test and excluded OOD files. It does not train, calibrate, or score a model.

`DATA_READINESS` remains `NO_GO_FOR_AUTHENTICITY_CLAIMS`. `ACQUISITION_AUTHORIZED` is false. `INFERENCE_AUTHORIZED` is false. Independently verified labels remain zero.

A record that validates against `manifest.schema.json`, or that names an evidence-record id, is not proof that a real label or source is trustworthy. Primary eligibility follows the Phase 55A fail-closed rules: verified brand, real retention date, `deletion_status` of `RETENTION_ACTIVE`, approved rights, independent evidence, a resolved label, and disjointness that is actually `PASS`. `UNKNOWN` is never converted to `PASS`.

Within-brand, cross-brand, and project-task-held-out-brand claims stay distinct and are not demonstrated here. The Phase 56 split contract and Phase 57/58 gap plans still apply. No source lead is approved.

Run the synthetic tests with the CI pin `jsonschema==4.26.0`:

```bash
python -m unittest discover -s ml_rtx5080/experiments/phase59_manifest_validator/tests -v
```
