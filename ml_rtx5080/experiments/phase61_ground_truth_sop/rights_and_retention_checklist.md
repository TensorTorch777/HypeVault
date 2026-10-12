# Rights and retention checklist

Use this list before setting `eligible_primary_benchmark` to true. Every line must be true for that record. A schema-valid pending record is not enough.

- [ ] The license is written, and `license_status` is `APPROVED`.
- [ ] The license names non-public ML evaluation. `permitted_purposes` contains `ML_EVALUATION`.
- [ ] `consent_status` is `APPROVED`, or a documented reason supports `NOT_REQUIRED`.
- [ ] `retention_until` is a calendar date, not `UNKNOWN`.
- [ ] `deletion_status` is exactly `RETENTION_ACTIVE`.
- [ ] Redisclosure and publication limits are written. Aggregate publication is allowed only if the grant says so.
- [ ] The rights holder is identified in the approved store, not in Git.
- [ ] Counsel has reviewed the handling terms for the named provider. WatchCSA remains on hold until that review replaces or accepts the published Member Content clause.
- [ ] No image, certificate, or private report has been copied into the repository.

This checklist does not itself approve any provider or record.
