# Protocol feasibility review v1

The frozen protocol file `watch_like_ood_protocol_v1.json` was not edited. Its hash remains `babaaf2b5c30d252c39765886ee5db032c09c4e3f25cf0e9eacc63f17128dde9`.

This review separates the current protocol from changes that were considered and not adopted.

## Why Phase 37 failed

Phase 37 qualified 10 sources, stored 1 file, and admitted 0. The registered package was not reached.

| Blocker | Evidence |
|---|---|
| Original bytes | 28 Commons originals were stored in Phase 36. 531 further size-eligible originals returned HTTP 429. Openverse returned Flickr `_b.jpg` renditions. Museums and Hippopx returned HTTP 410, 403, or 401. |
| Licensing | One CC0 original and one CC BY-SA original were identified. Neither could be admitted. A CC0 metadata database was not treated as a license for external photographs. |
| Provenance | The stored Pixnio file has a source URL and a license URL. WatchBase CDN links in Timepiecepedia do not. |
| Brand verification | 1 of 1 stored files has `brand_verification = unknown`. |
| Category verification | That file is a generic wristwatch page, not one of the four protocol groups. Phase 36 had already rejected 92 non-watch or part files and 6 boutique exteriors. |
| Byte cap | 456 of 1,146 Phase 36 failures were over the 1.5 MB cap or the wrong MIME type. The Flickr original was 2,475,875 bytes. |
| Access | 531 of 1,146 Phase 36 failures were HTTP 429. Access plus the byte cap account for 987 of 1,146 failures (86.1%). |
| Source independence | Admitted independent source groups: 0. The only category-verified scale is one site, Wikimedia Commons. |
| Derivatives | 1,648 local Roboflow `.rf.` files remain excluded. Openverse renditions were not downloaded. Phase 36 marked 36 derivative titles. |

The Commons files that failed only with HTTP 429 are an upper bound, not a corpus. If every one later passed license and content checks, the counts would be 258 unseen luxury, 155 ordinary, 104 smart/digital, and 14 wrist/store. The 40-image brand cap reduces Rolex from 82 category files to 40 and Tissot from 46 to 40, leaving about 216 unseen and 149 ordinary. Those ceilings miss 300, 200, and the wrist holdout of 25.

## Requirement classes

### Scientifically necessary

These change whether an escape rate is about real, independent, out-of-scope watches.

- The image is a real watch photograph, not a synthetic image, a diagram, or a movement-only part.
- Thumbnails, resized previews, screenshots, and `.rf.` augmentation exports are not substituted for originals.
- The five training brands stay out of the OOD set.
- Exact hashes, source URLs, and near-duplicate hashes do not overlap Phase 16, Phase 35, Phase 36, or the locked final test.
- Development images are not used to choose a threshold. The protocol forbids threshold search.
- A source family stays in one split.
- A conclusion requires 60 unseen-luxury, 40 ordinary, and 25 smart/digital holdout images, at least 4 unseen brands with at least 8 holdout images, and no single brand above 40% of the unseen holdout.
- The model under evaluation is not the brand verifier.

### Highly desirable

- Collection headroom of 300, 200, 100, and 50 images, so splits and rejections do not consume the holdout floors.
- More than one website and more than one photographer.
- A wrist/store group. The frozen protocol already says a wrist shortfall alone does not block a conclusion.
- The brand cap of 40, which is a diversity rule rather than a validity rule for a single rate.

### Operational constraints

- Inclusion text that names Commons original bytes as the collection path.
- The pre-listed Commons category table.
- MIME limited to JPEG, PNG, and WebP.
- The 1.5 MB ceiling.
- Rate-limit behavior of the Commons upload host.

### Potentially over-restrictive

- The 1.5 MB ceiling. Classification: **weakly justified**. See below.
- Treating a Commons category as the only acceptable brand declaration. A source-declared brand from another explicit record, or a documented expert check, can answer the same metadata question. Phase 37's CC0 file failed only this gap. Relaxing it was not adopted, because no licensed supply was shown to reach the sample-size floors after the change.

## 1.5 MB ceiling

The ceiling does not measure independence. A 972,467-byte Pixnio JPEG at 5400×3600 was an original under the cap. A 2,475,875-byte Flickr JPEG was an original over the cap. Both are camera-file class images. The scientific control is the rule that forbids resizing an original to sneak under the cap. The cap itself discarded 456 Phase 36 candidates and the Flickr pilot.

It is weakly justified as a disk and transfer constraint. It is not a scientific validity condition. It stays in force because this review does not edit the protocol.

## Source-declared brand

| Method | Assessment |
|---|---|
| A. Source-declared brand | Required for a named-brand rate. Commons categories are this kind of evidence. They are noisy, which is why title exclusions exist. |
| B. Expert or manual verification | Acceptable for a future version if the verifier, the visible evidence, and disagreements are recorded. It is not silent visual relabeling. |
| C. Multi-source verification | Acceptable when two independent records agree. A metadata row plus an external CDN image is not agreement about the photograph's license. |
| D. Model-assisted verification | The model under evaluation cannot verify its own OOD labels. No second model was justified as an oracle. This method is not available. |

Named-brand rates need A, B, or C. A binary claim that a watch is outside the five training brands needs a recorded exclusion, not a guessed logo. Phase 37 did not have that record for the CC0 pilot.

## Four routes

### Route A — licensed or open images

Not feasible for the registered package on the evidence in hand.

Commons is the only open source with category-verified brands at scale, and its ceiling misses the unseen, ordinary, and wrist targets. Pixnio proved that a CC0 original can be downloaded and still fail brand verification. Flickr proved that a CC original can exceed 1.5 MB. Openverse proved that an aggregator can return a resized file. Museum APIs were not reachable.

This route is not claimed to reach 60 unseen-luxury holdout images, 40 ordinary, 25 smart/digital, and 4 brands with 8 images each.

### Route B — newly captured photographs

No personal or private images were collected.

A researcher can photograph watches they own, borrow with permission, or photograph in a store with permission. Those files can be original, permissioned, and source-grouped. A realistic single-person set is a small number of watches and low tens of photographs. No collaborator program or store permission is on record. That scale does not evidence 60 unseen-luxury holdout images across 4 brands. The route can produce a clean illustrative set. It is not a demonstrated path to the registered confirmatory corpus.

### Route C — copyrighted photographs for local research

`LEGAL_BASIS = UNKNOWN`

Retailer, manufacturer, and marketplace photographs would be scientifically informative, because the pages often name the brand. No source license and no institutional approval were found. This route is classified `REQUIRES_INSTITUTIONAL_APPROVAL`. It is not a recommendation to proceed.

### Route D — existing research datasets

No large dataset was downloaded. The audited sets are Roboflow derivative exports, a CC0 metadata database with unknown external image rights, a local unlabeled archive, and a Hugging Face search for `wristwatch` that returned 0 datasets. None supplies independent original photographs with an explicit image license at the registered count.

## Science and governance

`SCIENTIFIC_FEASIBILITY = true`

A watch-like OOD measurement is a valid experiment. Phase 35 already ran one on 408 images, including 194 unseen-luxury images, and the Wilson interval for that unseen escape rate stays above 0.80. The scientific question is not unanswerable.

`DATA_GOVERNANCE_FEASIBILITY = false`

The registered confirmatory corpus cannot be assembled from images whose original bytes, brand or category evidence, and image license are all acceptable. Absence of that corpus is not evidence that the model is safe.

## Decision

`PROTOCOL_DECISION = SCOPE_REDUCTION_RECOMMENDED`

Protocol v1 is not operationally achievable for the registered package. A version-2 edit that only raised the byte cap, or that allowed expert brand labels, is not justified as a fix: the Commons ceiling still misses the targets, and no other licensed supply was shown to replace it. The open-set evidence package is blocked under acceptable governance. The supported description of the system is a five-brand authenticity research prototype.

Proposed changes, not adopted:

| Proposal | Scientific effect | Legal effect | Comparability |
|---|---|---|---|
| Raise or remove the 1.5 MB cap and keep the no-resize rule | Admits large originals that are the same kind of photograph | None by itself | A later corpus would be a new experiment. Phase 36 would stay frozen. |
| Allow a documented expert brand check | Could admit licensed photos that lack a publisher category | The image license is unchanged | Brand rates would need a `verification_method` field. |
| Admit thumbnails or `.rf.` exports | Would mix derivatives into the escape rate | Often a different file than the licensed original | Not comparable to an original-photo result. Not proposed for adoption. |
| Lower the holdout floors | Would repeat the Phase 36 mistake | None | Would make a weak sample look like a conclusion. Not proposed for adoption. |

## Phase 35 and Phase 36

Phase 35 remains `OOD_FAIL`. It is a real OOD experiment: 408 images, 194 unseen luxury, watch-like escape 0.8405 on 257 images, unseen escape 0.8814 on 194, Rolex 159/180 AUTHENTIC, Omega 10/12, Cartier 2/2. Its weaknesses are mixed sources (180 Roboflow, 188 Commons, 40 unknown-license local files), Rolex derivative concentration, thin brand cells outside Rolex, 18 wrist/store images, and some non-watch groups. Phase 35 is substantial warning evidence. It is not a final generalized open-set benchmark. It is not invalid.

Phase 36 remains `WATCH_OOD_UNRESOLVED`: 25 holdout images, 14 unseen luxury, 10 ordinary, 1 smart/digital, 0 wrist/store. It is an underpowered independent observation. It was not used to choose brands, sources, or thresholds.
