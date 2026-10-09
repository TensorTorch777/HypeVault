# Execution log — HypeVault Master Execution Plan

Run date: 2026-10-09. Each status rests on a command or record cited here.

Statuses: `PASS`, `FAIL`, `BLOCKED`, `IN_PROGRESS`, `NOT_STARTED`.

## Starting state (A1)

- `main` = `origin/main` = `e0c0c6d27afa5275909ff3c9a3fbe0713a28f330`. It did not move during this run.
- `feature/dual-model-triton` started at `ac17a87ff78397e2904366e05877fe2ca24dcef0`, matching its remote: one commit ahead of `main`, zero behind.
- No pull request existed (`gh pr list` returned `[]`). `main` has no branch protection, and the repository had no workflows.
- The working tree held 26 modified and 128 untracked paths from earlier phases. Unrelated work was left untouched: `ml_rtx5080/train.py`, `ml_rtx5080/regenerate_plots.py`, research modules, experiment folders, `.agents/`, and `skills-lock.json`.

## Commits made on `feature/dual-model-triton`

| Commit | Work item | Content |
| --- | --- | --- |
| `4b71af0` | A2, A3 | Ignore experiment weights; tracked byte-identical DINOv2 Triton config |
| `58945be` | A4, C2 | Customer labels, seed and import scripts, copy fixes, safety tests and their artifacts |
| `93ee6dd` | A6, C1, C3 | Research entitlement, public-only registration roles, DINOv2 adapter fixes, per-model readiness, research UI |
| `7c9501b` | A4, A5 | Non-interactive ESLint, CI workflow, `requirements-ci.txt` |
| `b639dc3` | B6 | Parity protocol and harness, committed before any result |
| `8cd2cab` | B, C4 | CPU Triton parity results, E2E script and results, runtime diagnostics |
| `f688ff7` | B2, E2, E3 | Admin GPU steps, data-governance workflow, record schema v2, admission validator |

## Workstream A — repository release readiness

| Item | Status | Evidence |
| --- | --- | --- |
| A1 | PASS | Git, remote, and PR state above |
| A2 | PASS | After `4b71af0`, `git ls-files --others --exclude-standard` shows 0 `.pt`/`.onnx`/`.onnx.data` files, and none are tracked. `epoch_018.pt` is ignored by `.gitignore:38`. Every staged diff was scanned for key and password patterns before commit, with no hits. |
| A3 | PASS | `infra/triton/dinov2_classifier/config.pbtxt` has SHA-256 `9488821b…7c4f`, byte-identical to the local config. The test reads the tracked copy. A separate local-only test checks the copies match and skips with a named reason in CI. |
| A4 | PASS | A clean `git archive HEAD` extract with no `.env` files runs 50 unit tests: OK, 2 skips naming the gitignored `models/dinov2_classifier` and the uncommitted `epoch_018.pt`. `npm run lint` reports 0 warnings, `tsc` passes, `next build` passes. 46 local research test files stay uncommitted because they need untracked research modules, about 105 GB of experiment artifacts, or the image catalog. They are not in CI. |
| A5 | PASS | CI run 37914776725 on `7c9501b`: backend CPU (50 tests, OK, skipped=2) and frontend jobs succeeded; the manual GPU job was skipped. Runs 37916597441 (push) and 37916603278 (pull_request) on `f688ff7`: 55 tests OK (skipped=2), lint 0 warnings, build compiled, both succeeded. |
| A6 | PASS | `UserRegister` accepts only `buyer` and `seller`, and the route re-checks `PUBLIC_REGISTRATION_ROLES`. `require_research_user` reads `RESEARCH_USER_EMAILS` on the server; empty means nobody. `tests/test_research_access_control.py` drives real FastAPI routing: anonymous gets 401, a logged-in non-entitled user gets 403 with no inference, an entitled user in production, unknown, or missing mode gets 403 `POLICY_ERROR`, and an `admin` role in registration is rejected. E2E repeated these against the live stack. |
| A7 | PASS | Draft PR [#1](https://github.com/TensorTorch777/HypeVault/pull/1), `feature/dual-model-triton` → `main`. The description lists scope, tests, serving blockers, data `NO_GO`, and the security fixes. |

## Workstream B — NVIDIA GPU runtime and Triton

| Item | Status | Evidence |
| --- | --- | --- |
| B1 | PASS | Raw output is in `docs/engineering/runtime_diagnostics_2026-10-09.txt`. The driver is not the cause: modules are loaded, `/dev/nvidia*` exists, and host PyTorch sees the RTX 5080 at compute capability 12.0. The causes are the container runtime/CDI (no toolkit packages, no CDI spec, no `daemon.json`), permissions (sudo needs a password), and, at that time, disk (915 MB free). |
| B2 | BLOCKED | Needs root. The exact official commands are in `docs/engineering/gpu_runtime_admin_steps.md`. Nothing was installed and the driver was not touched. |
| B3 | BLOCKED | Selected pin: `nvcr.io/nvidia/tritonserver:26.01-py3`, index digest `sha256:c9f2ede5…146b`, amd64 `sha256:8a4ecd6b…4be5`, 7.64 GB compressed. Release notes state driver 575 or later, Blackwell support, and ONNX Runtime 1.24.1. Not pulled: it cannot start on the GPU until B2 is done. Disk later rose to 42 GB from outside this work, so space is no longer the blocker. |
| B4 | BLOCKED | `dinov2_classifier` fails on Triton 23.10: `Unsupported model IR version: 10, max supported IR version: 9`. Separately, the live route runs local PyTorch `vit_base_patch14_dinov2.lvd142m` at 504, while `dinov2_classifier` is a 40-block ViT-g/14 at 518 with no identity file. Choosing which DINOv2 backs the selector is an owner decision. |
| B5 | BLOCKED | GPU execution is blocked. Done on CPU: the checkpoint SHA was re-verified, a single authoritative export (`model.onnx` SHA-256 `d1d0c9bc…3edb`, matching `identity.json`) was confirmed, and `dinov3_authenticity_candidate` v1 reached `READY` on CPU Triton 23.10 with explicit model control (the duplicate export was not loaded). |
| B6 | PASS (CPU scope) | Protocol `parity_protocol_v1.json` (SHA-256 `858abd0c…e4cb`) was committed in `b639dc3` before measurement. On CPU Triton vs. the FP32 CPU PyTorch reference, 24 non-test inputs (17 original sizes, train/validation/calibration) at batch sizes 1, 2, 4, 8 gave: max abs logit error `3.08e-5` (≤ `1e-4`), mean `3.21e-6`, max abs probability error `3.30e-7` (≤ `1e-5`), 0 decision mismatches. Batched and single outputs were bitwise equal, and server stats showed 112 inferences over 54 executions. DINOv2 parity is not run; its route is unavailable and returns `MODEL_ERROR` with `decision: null`. GPU parity is BLOCKED. |
| B7 | BLOCKED | Per the plan, metrics stay `null` while GPU execution is blocked (`latency_and_memory_results.json`). No CPU numbers were substituted. |

## Workstream C — dual-model application

| Item | Status | Evidence |
| --- | --- | --- |
| C1 | PASS | Only `dinov2_legacy` and `dinov3_experimental` are accepted; Triton names and paths get 422. Versions are pinned to `1`. There is no fallback: E2E flow 2 shows the DINOv3 inference count unchanged (1 → 1) when DINOv2 was selected and unavailable. Non-finite and wrong-size outputs fail closed. DINOv2 results carry `checkpoint_sha: null` and no DINOv3 calibration. |
| C2 | PASS | `/verify/authenticate` stays on the legacy model and rejects a DINOv3 selector (E2E flow 6). Research calls created 0 listings (flow 9). A legacy check stored `pending`, not `live` (flow 10). The 39 live listings show 19 × `Legacy screening — not verified` and 20 × `Demo listing — not verified` (flow 11). `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False`. |
| C3 | PASS | Browser check on the production build as an entitled user: DINOv2 — Legacy is disabled "(unavailable)", DINOv3 is selected and labelled "Experimental — Not approved for production", the red `EXPERIMENTAL — NOT APPROVED FOR PRODUCTION` notice shows, and each model is marked "not the live listing-check model". The result view shows `brand_verification = NOT_PERFORMED` and the model that ran, and discards a response from an unselected model. The file input cannot be driven by the browser tool, so the post-submit view was verified through the API response, not in the browser. |
| C4 | IN_PROGRESS | `scripts/e2e_dual_model.py` ran against Postgres, Redis, a research-mode API, a production-mode API, CPU Triton, and `next start`. All 13 flows passed; results are in `integration_test_results.json`. The E2E accounts and listing were deleted afterwards (53 listings and 8 users, same as before). Gaps: DINOv2 never served (flow 2 shows only fail-closed behaviour) and nothing ran on a GPU. |

## Workstream D — merge and release

| Item | Status | Evidence |
| --- | --- | --- |
| D1 | IN_PROGRESS | The PR targets current `main`, and CPU CI is green. The diff contains no weights, `models/`, `.env`, or images. GPU parity is explicitly blocked in the PR description. The selector disables unavailable models. |
| D2 | BLOCKED | Merging needs human review and approval of the staged release, plus the DINOv2 identity decision. No approval exists. The agent did not merge. |
| D3 | BLOCKED | Not merged, so there is no merge commit to verify. |

## Workstream E — data governance

| Item | Status | Evidence |
| --- | --- | --- |
| E1 | PASS | No training, relabelling, calibration, or threshold change. The final-test images were not opened: the parity and E2E inputs come from the non-test parity protocol, and both scripts refuse `split_manifest_v2` test paths. |
| E2 | PASS | `docs/data_governance/` holds the outreach message, the contribution and research-use permission draft (marked as needing legal review, not legal advice), the adjudication form, the disputed-label policy, the consent and licence record process, and the storage and retention policy. No permission or partner exists. |
| E3 | PASS | `candidate_image_record_v2.schema.json` extends `image_label_evidence_v1`. `ml_rtx5080/data_admission.py` admits a record only with real evidence; UNKNOWN or forbidden evidence (directory name, model prediction, wordmark only, JPEG settings, and similar) is rejected. `tests/test_data_admission.py`: 5 tests OK. |
| E4 | BLOCKED | No qualified examiner or partner, and no legal data-use permission. `DATASET_READY_FOR_TRAINING = false`. |

## Workstream F — scientific research

| Item | Status | Evidence |
| --- | --- | --- |
| F1–F5 | BLOCKED | Gated on E4. Nothing started. |

## Workstream G — production readiness

| Item | Status | Evidence |
| --- | --- | --- |
| G1 | BLOCKED | Neither model is loaded on the target GPU. DINOv2 is not served. Load, latency, and memory are not measured. CPU parity, fail-closed behaviour, and per-model readiness do pass. |
| G2 | BLOCKED | No independent label provenance (data `NO_GO`). Phase 35 OOD failure stands. |
| G3 | IN_PROGRESS | Research route authorization, registration role limits, declared-brand wording, and publication separation pass. Rollback and version selection have not been tested on a real serving stack. No approval is recorded. |
| G4 | BLOCKED | Promotion is a human decision after G1–G3 pass. `DINOv3_PRODUCTION_ALLOWED = false`. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = false`. The OOD threshold stays null. Automatic publication stays blocked. |

## Local suite

`.venv/bin/python -m unittest discover -s tests -v`: 433 tests, OK (skipped=3: Triton not on port 8001, ONNX CUDA provider libraries missing, TensorRT/Triton unavailable), 48.4 s.

An earlier run had one error in `test_tiny_training_smoke`, which trains a tiny model on synthetic data. It writes a 934 MB temporary checkpoint and hit a full disk. After the 9 MB `/tmp` extract and the 164 MB `frontend/.next` build output were removed, it passed.

## Environment changes made

- Frontend devDependencies `eslint@8.57.1` and `eslint-config-next@14.2.5` (lockfile updated).
- `models/dinov3_authenticity_candidate` (gitignored) was set up in Phase 50 with symlinks; it is unchanged here.
- Containers `hypevault-postgres` and `hypevault-redis` were started for E2E and stopped afterwards. Triton CPU containers ran read-only and were removed.
- No drivers, system packages, Docker configuration, model weights, calibration values, or thresholds changed.
