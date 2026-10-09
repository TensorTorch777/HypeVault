# Publication gate

`listings.verdict` stores the model screening result. `listings.status` stores publication. `POST /verify/authenticate` still runs legacy DINOv2 and still records that verdict. It no longer sets `live`.

An `AUTHENTIC` result leaves the listing `pending`. A `FAKE` result sets `rejected`. `can_publish_listing` returns false while `AUTHENTICITY_MODEL_PRODUCTION_APPROVED` is false, including for the DINOv3 research prototype. The seller update schema cannot set `status` or `verdict`.

The legacy route remains reachable for the existing listing flow. It is labeled legacy and cannot publish from the model result.
