# Customer boundary

`POST /verify/authenticate` stays the legacy DINOv2 listing route. A caller needs an access cookie or bearer token. There is no rate limit. The route can still write a listing verdict. Its response now says `LEGACY_DINOV2`, `research_candidate = false`, and `production_validation = NOT_ESTABLISHED`. It does not carry the DINOv3 checkpoint or temperature.

`POST /research/verify` is the frozen DINOv3 prototype. It does not create a listing. It runs only in research or shadow mode. Production, missing, and unknown modes return `POLICY_ERROR` with `decision = null`. A checkpoint or temperature mismatch does the same and does not score.

The seller upload flow is labeled as the legacy check. The research demo is a separate page.
