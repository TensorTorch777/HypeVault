# Research demo hardening

Phase 40 changes the product contract around the frozen five-brand research prototype.

- Declared brand and image-level brand verification are separate. Only the declaration exists.
- Unsupported, missing, partial, and unknown brands stay fail-closed.
- The live HTTP classifier remains DINOv2. The DINOv3 candidate stays on the shadow path and is blocked when `HYPEVAULT_DEPLOYMENT_MODE` is production, which is the default.
- Research requests log mode, live model identity, declared brand, scope version, policy version, the research-candidate checkpoint SHA, status, and decision. Image bytes are not logged.
- Customer-facing copy describes a five-brand research prototype and states that the selected brand is not independently verified from the image.

The frozen checkpoint, temperature, calibration, final test, and live DINOv2 files were not modified. Promotion remains forbidden.

Frontend smoke on 2026-10-07: the Next.js dev server started on port 3000. The homepage title, declaration warning, Rolex/Omega “Outside scope” labels, and DINOv3 shadow wording were visible. `/seller/upload` stopped at login. PostgreSQL was not listening on port 5432 and the API was not listening on port 8000, so supported-brand classification, an unsupported submission, a missing brand, an invalid image, and the authenticated result page were not exercised in the browser.
