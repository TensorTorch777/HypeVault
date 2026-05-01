# Image Upload Skill

When building image upload flows:

- Support multi-image upload with drag-drop and preview
- Enforce client-side validation: minimum 800x800px for seller listing flows; server may enforce 400x400 minimum where specified
- Show upload progress and per-file error states
- Prefer direct-to-S3 via pre-signed URLs for production
- Never trust client-only validation alone — re-validate dimensions and MIME on the API
