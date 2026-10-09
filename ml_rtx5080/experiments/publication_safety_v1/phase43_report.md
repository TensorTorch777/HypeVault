# Phase 43

The local database was read. It has 53 listings: 39 live, 1 pending, and 13 rejected. There is no publication timestamp, no model identifier, and no publication-decision column. Every live row is `PUBLICATION_TIME_UNKNOWN`.

Twenty live rows match the historical seed writer. Nineteen match the old authenticate route, which stored an image and a model verdict and, before Phase 42, set `live` for a non-fake verdict. None were modified.

`AUTHENTICITY_MODEL_PRODUCTION_APPROVED` remains false. A rolled-back fixture stored `AUTHENTIC` and stayed `pending`. The live counts were the same before and after that fixture.

Existing live listings are historical product state and are not automatically reclassified as authentic or fake by this phase.
