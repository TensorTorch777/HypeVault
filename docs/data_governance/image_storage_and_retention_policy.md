# Image storage and retention policy

## Storage

- **Where images live:** original bytes go to a dedicated access-controlled store. Do not use the git repository, the application upload folder, or a public bucket.
- **Access:** read access is limited to named researchers. Every access is logged.
- **Immutability:** originals are write-once, and each record's SHA-256 is computed at ingestion. Derivatives (resized copies, crops) live in a separate location. They reference the original SHA-256 and carry `derivative_of`.
- **Evidence and identity:** examiner identities and evidence documents are stored apart from the images, under access-controlled IDs. Training data and manifests contain only those IDs.
- **Personal data:** images showing faces, documents with personal details, or location metadata are redacted at ingestion. EXIF GPS is stripped and the stripping is recorded.

## Retention

- Each image is kept until the earliest of: its agreement's retention end date, a withdrawal, or the end of the research purpose.
- **Deletion:** remove the original, every derivative, and every cached copy. Then record the deletion date in the manifest history. Model weights trained on deleted images are handled as the agreement's Schedule C states.
- **Reviews:** retention dates are checked before each new manifest version.

## Separation from the existing catalog

The current 30,000-image catalog stays quarantined (Phase 48). New verified images are never mixed into its folders and never inherit its directory labels.
