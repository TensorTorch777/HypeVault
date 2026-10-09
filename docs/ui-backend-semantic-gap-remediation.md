# HypeVault — Frontend/Backend Semantic Gap Remediation

**Prepared:** 2026-10-09  
**Purpose:** Remediation brief for Grok 4.7  
**Suggested implementation branch:** `fix/ui-backend-semantic-gaps`  
**Base branch:** `main`  
**Scope:** Customer-facing copy, UI states, and frontend/backend flow consistency before Phase 52. This document is a requirements brief, not evidence that any code changes have already been made.

---

## 1. Goal and product truth

HypeVault is currently a **research-prototype screening experience**, not a general-purpose or independently validated watch-authentication service. The UI must communicate what the application actually knows, what the model returned, and what the marketplace workflow did—without merging those three things into one claim.

Treat these as non-negotiable product truths throughout the application:

1. **The selected brand is user-declared.** The backend scope gate checks the supplied brand against the supported scope; it does not identify or independently verify the brand from the image.
2. **A model output is not proof of authenticity.** Neither DINOv2 nor DINOv3 currently provides a verified authenticity certificate or a scientifically established real-world guarantee.
3. **Model serving readiness is not model validity.** A model can be available to serve requests while its real-world authenticity performance remains unestablished.
4. **Listing visibility is not authenticity.** A visible/live listing means the listing is visible in the marketplace. It does not mean the item is authentic or independently verified.
5. **DINOv2 and DINOv3 are separate lanes with different status.** DINOv2 is the legacy path. DINOv3 is experimental/research-only and is not approved for production authenticity decisions.
6. **Mock/demo content must not look like live product data.** Static example listings, price points, and UI mockups must be visibly identified as examples—or replaced with real API-backed data and honest freshness/source metadata.
7. **Unsupported scope is not a negative authenticity result.** If a brand is out of scope or a request is rejected, explain that the system cannot provide a result for that request. Do not imply the product is fake.
8. **Publication remains blocked by the current safety gate.** Frontend changes must not create a path where model output authorizes publication.

### User-facing terminology

Prefer precise, plain-language distinctions:

| Concept | Preferred wording | Avoid implying |
|---|---|---|
| Brand input | **Declared brand** — selected by the user; not verified from the image | “Detected brand” or “verified brand” |
| Model response | **Research screening result — not verified** | “Authenticity confirmed” / “certificate” |
| Model status | **Serving available/unavailable** | “Accurate,” “valid,” or “trustworthy” based only on readiness |
| Listing status | **Visible**, **Pending**, or **Rejected by screening workflow**, with explanation | “Live means authentic” |
| Static chart/listing | **Illustrative example / demo data — not live market data** | “Live inventory,” “current asks,” or equivalent without live evidence |
| Unsupported brand | **Outside current research scope — no result** | “Fake,” “rejected as counterfeit,” or any authenticity verdict |

---

## 2. Main semantic gaps

Priority meanings:
- **P0:** Must be addressed before the team starts Phase 52; the issue could materially mislead users about trust, authenticity, or live data.
- **P1:** Required for reliable end-to-end flow and clear error/recovery behavior before treating the UI audit as complete.
- **P2:** Useful follow-up polish; do not let this expand into a redesign or delay the safety-critical fixes.

| ID | Priority | Surface / code area | Gap | Required change | Acceptance condition |
|---|---|---|---|---|---|
| SG-01 | P0 | Homepage hero and marketing copy — `frontend/src/components/hero/PremiumHero.tsx`, `frontend/src/app/page.tsx`, `PhilosophyBento.tsx`, `ClosingCTA.tsx` | “Trust” and “evaluated brands” positioning may sound stronger than the evidence supports, given the known dataset provenance and confounds. | State consistently that this is a **research prototype** for in-scope watch-brand screening. Do not claim real-world accuracy, verified authenticity, or an independently validated five-brand benchmark. | A new visitor can tell from the homepage that this is a research prototype and not a certificate/authentication guarantee. |
| SG-02 | P0 | “How it works” — `frontend/src/components/home/HowItWorks.tsx` | The animated scan grid/targets may look like actual model localization or a live inference visualization even though the graphic is hard-coded/illustrative. | Keep a visible label adjacent to the mockup: **“Illustration only — not a live scan or model-generated localization.”** Do not describe the target labels as model detections. | The visualization cannot reasonably be mistaken for a real prediction overlay when viewed without surrounding explanatory copy. |
| SG-03 | P0 | Home inventory — `frontend/src/app/page.tsx` | `GENERIC_RECENT_ITEMS` contains hard-coded demo examples, while section wording includes “Inventory · live” / “FRESH IN THE VAULT.” | Either wire the section to the actual recent-listings API and represent request failure separately, or clearly label the existing cards as **demo examples — not live inventory**. Do not silently present static items as live listings. | No hard-coded card is labelled or framed as live inventory. If data is API-backed, each visible item originates from the response. |
| SG-04 | P0 | Market/price promotional mockup — `frontend/src/components/home/HowItWorks.tsx`, `AppleBentoPromo.tsx` | Static example prices/series can be read as real current marketplace asks or price intelligence. | Either use a real data source with source and last-updated/freshness context, or mark the graph and prices as **illustrative sample data — not live market prices**. Do not imply live Chrono24/StockX/eBay collection unless the backend actually returns those observations. | Every static price visual is clearly labelled as illustrative; no text claims live comparison unless verified by the implementation. |
| SG-05 | P0 | Research result — `frontend/src/app/research/page.tsx`, `frontend/src/components/AuthBadge.tsx` | A categorical `AUTHENTIC` / `FAKE` response can be mistaken for ground truth, and “Rejected” may be mistaken for an independently proven counterfeit finding. | Label the model output as a **model classification/screening result, not verified**. Explain separately what the workflow did (for example, pending or rejected by the legacy screening workflow). Preserve the model's raw response for diagnostics; improve display wording instead of falsifying or silently rewriting the backend output. | Users can distinguish model output from a verified fact and from the listing workflow status. |
| SG-06 | P1 | Brand input and scope messages — research, seller upload, product detail | “Brand” may be understood as image-identified even though the brand is user-declared and passed into the backend scope gate. | Label it “Declared brand” and place a short, persistent explanation by the selector/result. In the unsupported-scope state, say no result was produced and the reason is scope—not authenticity. | Brand source and scope limitation are apparent before submission and on the result/error state. |
| SG-07 | P1 | Research selector — `frontend/src/app/research/page.tsx` | Readiness can be interpreted as scientific approval or correctness. | Name operational readiness **“Serving status”** (available/unavailable). Separately show: **“Real-world authenticity validity: not established.”** Maintain the explicit DINOv3 experimental/not-production-approved status. | The UI does not equate successful model loading with accuracy, authorization, or production approval. |
| SG-08 | P1 | Seller upload — `frontend/src/app/seller/upload/page.tsx` | The flow creates a listing before requesting the legacy screening result. If the follow-up request fails, users need a clear, recoverable state rather than a vague failure that invites duplicate submission. | Represent the stages separately: listing created, screening request in progress, screening response received/failed. On failure, retain the created listing ID, explain that its state may be pending, and offer a safe retry of the screening request without creating a second listing. Do not claim successful verification. | Simulated API failure after listing creation does not lose the listing reference, falsely report a result, or create a duplicate when screening is retried. |
| SG-09 | P1 | Listing pages / status badges — `customerLabel.ts`, `ProductVisualPanel.tsx`, `frontend/src/app/product/[id]/page.tsx` | Some surfaces correctly explain visibility versus authenticity, but the meaning can be lost when the same status appears in a card, badge, or detail page. | Make the visibility-versus-authenticity distinction consistent anywhere a listing status is shown. Keep “Published/Visible — not verified” semantics clear on card and detail surfaces. Explain that a legacy rejection is a workflow state, not independent proof of counterfeit status. | Listing card, badge, and detail page never imply that `live`/visible means authentic. |
| SG-10 | P1 | API wrapper and data states — `frontend/src/lib/api.ts` and related pages | `fetchRecentListings` can collapse a failed request into an empty list, and price/comparison helpers may return placeholder payloads with an `unavailable` error. A user may see “no results” instead of “request failed.” | Model **loading**, **success with data**, **success with no data**, and **error/unavailable** as distinct UI states. Never render placeholder comparison data as a valid comparison. Add retry where appropriate. | A mocked network failure displays an error/unavailable state; a valid empty response displays an empty state; these states are distinguishable. |
| SG-11 | P1 | Research authorization errors — research page and API wrapper | Research access is controlled separately from public buyer/seller roles; a 401/403 may be confusing if shown as a generic inference failure. | Show a concise authorization message for unauthorized access, distinct from model unavailable, invalid input, unsupported scope, and inference errors. Do not expose internal allowlists, secrets, or sensitive diagnostics. | Each failure category has a clear, safe user-facing message and no misleading model result. |
| SG-12 | P2 | Model comparison experience | DINOv2 and DINOv3 have different preprocessing and decision policies, and their outputs should not be presented as perfectly comparable truth scores. | If comparison is displayed, make it an explicitly research-only side-by-side comparison, show selected model and its lane, do not combine outputs into a single verdict, and explain that scores are model-specific and not proof of authenticity. | No blended “winner” or combined authentication verdict is shown; DINOv3 is still labelled experimental. |

---

## 3. The most important frontend–backend mismatches

These are cross-layer contract gaps, rather than just copy edits.

| ID | Frontend can suggest… | Backend/source-of-truth currently says… | Change required |
|---|---|---|---|
| G-01 — Illustration vs inference | Scan lines, boxes, or named watch parts are generated by the model for the submitted image. | `ScanMockup` is a static/animated illustration; it is not the inference output or evidence of localization. | Label the visualization persistently as illustrative; avoid words such as “detected” for its hard-coded annotations. |
| G-02 — Brand identification | The app identified the brand from the photo. | `backend/inference/scope_gate.py` validates a caller-provided brand against the supported scope; it does not identify the brand from image pixels. | Say **declared brand** across the input, result, and listing surfaces. |
| G-03 — DINOv2/DINOv3 equivalence | Both models are interchangeable production authenticators. | `backend/inference/model_router.py`, `research_routes.py`, and `routes.py` route distinct model policies. DINOv3 is experimental and research/shadow-only; the legacy endpoint is DINOv2-only. | Keep lanes visibly separate; explain model identity and limitation on every relevant result. Do not expose a DINOv3 choice on a production-authentication path. |
| G-04 — Readiness vs validity | A model shown as ready is approved or proven accurate. | Readiness is an operational serving/runtime signal. Known data provenance/class-geometry confounds mean authenticity performance is not scientifically established. | Rename readiness to “Serving status”; display validity/approval separately as “not established” / “not approved for production.” |
| G-05 — Model output vs marketplace action | A `FAKE`/`AUTHENTIC` label establishes the true status of a watch or is the same thing as the listing status. | `backend/inference/publication_gate.py` keeps production approval false; legacy fake output maps to a rejected workflow status and an authentic output remains pending. Neither is independent verification. | Render the model classification and workflow effect as separate fields. Explain what “Pending” or “Rejected” means in the legacy workflow. |
| G-06 — Publication/live status | “Live” / “Published” means authentic and certified. | `backend/listings/routes.py` exposes live listings; the publication gate blocks model-authorized publication. Visibility and authenticity are different concepts. | Use “Visible in marketplace — authenticity not verified” or equivalent anywhere status could be misread. |
| G-07 — Hard-coded data vs live APIs | Homepage inventory and prices are current API results. | `GENERIC_RECENT_ITEMS` and the `PriceChartMockup` values are hard-coded examples. | Connect to a genuine source or explicitly label as demo/illustrative. Never make an empty/error response look like real zero inventory or a real market quote. |
| G-08 — Failure vs no data | An empty list/comparison means nothing is available. | API wrappers can return empty/placeholder objects on failure. | Preserve an explicit status/error field through the UI; distinguish request failure, empty success, unavailable provider, and loaded data. |
| G-09 — Out-of-scope vs counterfeit | Brand out of scope means the image is fake. | The scope gate returns an unsupported-scope/no-verdict response; it does not make an authenticity judgement. | Use “outside current scope — no result,” not a fake/rejected authenticity verdict. |
| G-10 — Research authorization vs normal app account | Any signed-in buyer/seller can use the research selector, or a generic inference error means the model failed. | Research access uses a separate authorization dependency. Ordinary buyer/seller roles are not automatically research authorization. | Give 401/403 a safe, clear access message and separate it from model/runtime errors. |

---

## 4. Gaps in the actual application flow

### 4.1 Homepage and discovery

**Current risk:** The homepage mixes a research prototype message with polished marketplace language, hard-coded example cards, and static pricing visuals. That can cause visitors to believe the inventory and price intelligence are live or that the model has validated the items.

**Required behavior:**
- Every static inventory card/price visual is explicitly marked **demo/illustrative** OR is fed from a real data response with appropriate source and freshness labels.
- “Five evaluated brands” must not overstate evidence. Prefer “five in-scope brands configured for the research prototype” unless the displayed evaluation claim is separately substantiated.
- The “How it works” scan artwork has a visible illustration disclaimer in the visual itself.
- A failed data request must not quietly render as an empty market/inventory result.

### 4.2 Research model selection and submission

**Current risk:** A healthy/ready model can be mistaken for a scientifically approved one. The selected brand may be mistaken for image-derived brand identity. A categorical prediction can read like a certificate.

**Required behavior:**
- The selected model name and lane remain visible during submission and on the result.
- DINOv2 is identified as the legacy research-prototype path; DINOv3 is identified as experimental and not approved for production.
- “Serving status” is separate from “real-world authenticity validity / approval.”
- “Declared brand” is visible at the selector and result, alongside “not independently verified from image.”
- Success text describes a **model screening/classification result**, not a verified fact.
- Unsupported scope, invalid image, access denied, service unavailable, and inference error are distinct states; none should be fabricated into an authenticity result.
- Continue rejecting a response whose model identifier does not match the selected model. Preserve the existing fail-closed behavior.

### 4.3 Seller upload and legacy screening

**Current risk:** Listing creation and screening are separate requests. A screening/network failure after creating the listing must not leave the user unsure whether the listing exists or encourage a duplicate listing.

**Required behavior:**
1. Show a clear “Creating listing” state.
2. Once creation succeeds, retain the resulting listing ID and status.
3. Show a separate “Legacy screening in progress” state while `/verify/authenticate` runs.
4. If screening succeeds, display the model result and listing workflow state as **separate facts**.
5. If screening fails after creation, state that the listing was created but screening did not complete; display its pending/unverified status and allow a screening retry using the existing listing ID.
6. Do not retry listing creation automatically when only the screening request failed.
7. Do not claim the listing is authenticated or published by the screening response.
8. Unsupported brands should yield a clear “outside scope / no result” message, not a fake label.

### 4.4 Listing card and detail page

**Current risk:** A status badge viewed without surrounding text can be interpreted more strongly than intended.

**Required behavior:**
- Keep visible/published status separate from authenticity statements.
- Keep “Declared brand” terminology on details.
- Use one shared customer-facing label/helper where practical so that cards, badges, and detail pages cannot drift semantically.
- A legacy rejection must be explained as a screening/workflow outcome and not independent proof of counterfeiting.
- Never label the item “verified authentic” based on model output.

### 4.5 Market-price comparison and recent listing data

**Current risk:** Static example values or API failure placeholders may look like live market facts.

**Required behavior:**
- For real API data, show source and freshness when the backend exposes them; do not invent source attribution or timestamps.
- For mock data, mark it on the component itself as illustrative.
- Treat `error: unavailable` as an error state, not as a successful quote or an empty comparison.
- Distinguish “no data returned” from “could not load data.”
- Include retry only where retry is safe and useful.

### 4.6 Research authorization and backend error mapping

**Current risk:** A general error toast can hide the actual cause and imply an inference failure when access is denied or the brand is out of scope.

**Required behavior:** Map response categories to honest, safe user messages:

| API condition | User-facing state | Must not say |
|---|---|---|
| 401/403 research access denied | Research access is not enabled for this account/session | “Model failed” or show a result |
| Unsupported brand/scope | Outside current research scope; no result produced | “Fake” / “counterfeit” |
| Invalid image or input | Image could not be processed; ask for a supported file | “Fake” / “authentic” |
| Model/runtime unavailable | Screening temporarily unavailable; no result produced | Return a cached/fallback verdict silently |
| Network timeout after listing creation | Listing exists in pending/unverified state; screening did not complete | “Upload failed” if the listing was actually created |
| Successful model inference | Research screening result, not verified; separate listing status | “Authenticity confirmed” |
| Empty API success | No items/results currently available | “Service unavailable” unless it actually failed |
| API failure | Could not load data; retry when safe | “No items found” |

---

## 5. What to prioritize before Phase 52

### Phase A — Mandatory trust and truthfulness fixes (P0)

- [ ] Remove or qualify homepage language that sounds like proven real-world authentication or a validated benchmark.
- [ ] Clearly label hard-coded homepage listing cards as demo examples or switch them to the real API with explicit loading/error/empty states.
- [ ] Clearly label all static price/market visuals as illustrative, unless genuinely API-backed with source/freshness evidence.
- [ ] Put a visible “illustration only; not a live scan or model-generated localization” label directly on/adjacent to the animated scan mockup.
- [ ] Rewrite research results to separate **model classification**, **declared brand**, **not independently verified**, and **listing workflow status**.
- [ ] Ensure no customer-facing surface implies that a visible/live/published listing is authentic.

**P0 exit gate:** A user cannot reasonably interpret the homepage, price mockups, scan artwork, or result badge as proof of real-world watch authenticity or as live data when it is static.

### Phase B — Flow and reliability fixes (P1)

- [ ] Rename model “readiness” displays to “Serving status”; explicitly state that real-world authenticity validity is not established and DINOv3 is not production-approved.
- [ ] Use “Declared brand” consistently in research, seller upload, and product detail experiences.
- [ ] Separate unsupported scope, access denial, invalid input, model unavailable, network failure, successful result, and empty data into different states.
- [ ] Preserve distinction between listing creation and screening success; make screening retry use the existing listing ID.
- [ ] Audit all listing badges and detail surfaces for consistent “visible/published does not mean verified” wording.
- [ ] Make API errors distinguishable from empty results; remove placeholder success-looking data from error paths.
- [ ] Add focused UI/API tests for the states above and verify them in the browser where available.

**P1 exit gate:** The user can always understand whether data loaded, whether the model ran, what the model returned, whether the item/listing exists, and whether any independent verification occurred (currently, it did not).

### Phase C — Safe polish (P2, only if quick)

- [ ] Add a concise research-only explanation if side-by-side model comparison is shown.
- [ ] Ensure responsive layouts do not hide key disclaimer text or put it only in hover tooltips.
- [ ] Add accessibility checks for visible statuses, error messages, and loading state announcements.

Do not expand this into a full redesign. Complete P0 and the practical P1 items first.

---

## 6. Likely code areas to inspect

Verify current code before editing; paths may have changed since this audit.

### Frontend

- `frontend/src/app/page.tsx`
- `frontend/src/components/hero/PremiumHero.tsx`
- `frontend/src/components/home/HowItWorks.tsx`
- `frontend/src/components/home/PhilosophyBento.tsx`
- `frontend/src/components/home/AppleBentoPromo.tsx`
- `frontend/src/components/home/ClosingCTA.tsx`
- `frontend/src/app/research/page.tsx`
- `frontend/src/app/seller/upload/page.tsx`
- `frontend/src/app/product/[id]/page.tsx`
- `frontend/src/components/AuthBadge.tsx`
- `frontend/src/components/product/ProductVisualPanel.tsx`
- `frontend/src/lib/customerLabel.ts`
- `frontend/src/lib/api.ts`

### Backend contracts to verify; change only when actually necessary

- `backend/inference/scope_gate.py`
- `backend/inference/research_routes.py`
- `backend/inference/model_router.py`
- `backend/inference/routes.py`
- `backend/inference/schemas.py`
- `backend/inference/publication_gate.py`
- `backend/inference/research_guard.py`
- `backend/listings/routes.py`
- `backend/listings/models.py`
- `backend/main.py`

Prefer fixing presentation/state handling in the frontend when the backend contract already expresses the correct semantics. If a contract change is needed, document the reason, preserve backwards compatibility where practical, and add tests.

---

## 7. Required tests and verification

Add or update tests appropriate to the existing framework. Do not claim tests passed unless they were run.

### Minimum acceptance checklist

- [ ] Hard-coded homepage listing examples are explicitly labelled as demo, or are replaced by actual returned records.
- [ ] Static price mockups have an in-component “illustrative / not live market data” label.
- [ ] Static scan mockup clearly says it is not model-generated localization or a live scan.
- [ ] Research selector uses “Serving status” and separates availability from validity/production approval.
- [ ] Result UI displays declared brand and “not verified” context with enough visibility on mobile and desktop.
- [ ] A fake/authentic model output is visually separated from the listing status.
- [ ] Out-of-scope, unauthorized, invalid-image, unavailable-model, timeout, empty-success, and non-empty-success flows have distinct messages.
- [ ] Recent-listing fetch failure does not masquerade as an empty list.
- [ ] Comparison API failure does not masquerade as a valid price comparison.
- [ ] Screening failure after listing creation preserves the listing ID and permits screening retry without duplicating the listing.
- [ ] A visible/live/published listing never receives “verified authentic” copy from this workflow.
- [ ] No DINOv3 selection or result is presented as production-approved.
- [ ] Publication remains blocked; model output alone cannot publish a listing.
- [ ] Existing API/model identity mismatch guard remains in place.
- [ ] Type checks, lint, relevant unit/integration tests, and production build are run where the repository supports them; report any command that could not run and why.

### Manual browser pass

Exercise the homepage, research page, seller upload, listing card/detail, and price comparison at desktop and mobile widths. Verify that disclaimers are visible without hover and error/empty/loading states are understandable. Use controlled/mocked failures for testing; do not alter production safety flags to force a scenario.

---

## 8. Hard guardrails — do not change as part of this task

This task is **UI/API semantic remediation only**, not a model-validation or model-promotion task.

**Do not:**

- Train, fine-tune, retrain, recalibrate, or change model weights.
- Change DINOv2 or DINOv3 checkpoints, SHA-256 digests, temperatures, thresholds, preprocessing, precision, Triton configurations, or model-router identities.
- Touch, open, copy, score, or otherwise access the sealed locked final test set.
- Use the known folder-derived catalog labels to claim real-world authenticity accuracy.
- Change production-approval flags or enable DINOv3 production inference.
- Weaken authorization, scope gates, fail-closed behavior, response/model identity checks, research/shadow deployment checks, or publication gates.
- Make listing publication depend on `AUTHENTIC`/`FAKE` model output.
- Introduce a silent model fallback or present fallback output as the selected model's result.
- Convert unsupported scope, access denied, unavailable inference, or parsing errors into a fake/authentic verdict.
- Start Phase 52, change its protocol, or create Phase 52 artifacts while implementing this brief.
- Rewrite the unrelated research branch or mix UI changes into `research/dinov3-cross-brand-v2`.

If an apparently necessary UI fix requires changing a safety invariant, **stop and report the conflict** instead of changing the invariant.

---

## 9. Expected deliverable from Grok 4.7

1. Implement this brief on `fix/ui-backend-semantic-gaps` (or report if the branch cannot be created).
2. Keep the diff focused. Avoid unrelated refactors and redesigns.
3. Update/add tests for the touched UI state and API behavior.
4. Run the relevant checks and report exact commands/results, including failures or checks not run.
5. Provide a compact completion report with: files changed, key behavior changes, tests, remaining gaps, and confirmation that the guardrails above were preserved.
6. Do not merge the branch automatically. Leave the changes reviewable for human approval.

---

## 10. Prompt for Grok 4.7

> Read `docs/ui-backend-semantic-gap-remediation.md` completely and implement it as written. First inspect the current repository state and branch list. Create and switch to `fix/ui-backend-semantic-gaps` from the latest `main`; do not work on `research/dinov3-cross-brand-v2` or any Phase 52 branch. If branch creation is blocked, stop before modifying source files and report the blocker.
>
> Implement the P0 fixes first, then the practical P1 fixes. Keep this scoped to customer-facing semantics, UI state/error handling, and safe seller-flow recovery. Add/update tests, run the available checks, and report exact outcomes. Do not claim a check passed unless you ran it.
>
> Follow every hard guardrail in the brief. Do not train or modify models, checkpoints, thresholds, temperatures, preprocessing, Triton/model routing, production-approval flags, scope/access/publication gates, or any locked-test data. Do not start Phase 52. Do not silently fall back to a different model. Do not merge or push to `main`. Finish with a clear summary of the changed files, behavioral outcomes, tests, and unresolved issues.
