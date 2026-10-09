# HypeVault implementation roadmap

Evidence date: 2026-10-09. Every status below comes from a check run on that date. Nothing here is a promise that data, hardware, or approvals will become available.

This is the pre-execution snapshot. Current status per work item is in [execution_log.md](execution_log.md), and the outcome is in [final_execution_summary.md](final_execution_summary.md).

## Verified state

### Git and GitHub

- `main` and `origin/main` are both `e0c0c6d27afa5275909ff3c9a3fbe0713a28f330`. It has not moved.
- `feature/dual-model-triton` and its remote are both `ac17a87ff78397e2904366e05877fe2ca24dcef0`. The branch is one commit ahead of `main` and zero behind.
- No pull request exists. The recorded URL is GitHub's "create pull request" link, not a pull request.
- `main` has no branch protection (GitHub API returns `Branch not protected`). The repository has no GitHub Actions workflows. Nothing requires review or CI before a merge.
- The working tree still holds 26 modified and 128 untracked paths from earlier phases. None are staged.

### Clean-checkout behaviour of the pushed commit

These checks ran on a `git archive` extract of `ac17a87`, outside the working tree.

- The backend imports, and it registers `/verify/authenticate`, `/research/verify`, and `/research/models`.
- The frontend passes `tsc --noEmit`, using the local `node_modules`.
- 14 tests ran and 1 failed. `test_candidate_config_does_not_replace_dinov2` reads `models/dinov2_classifier/config.pbtxt`, and `models/` is gitignored.
- Only 2 of the 48 local test files are committed. The 416-test local run depends on untracked modules and experiment artifacts, so it is not reproducible from Git.
- `next lint` has no ESLint config and stops at an interactive setup prompt. `next build` passes locally.

### Runtime

| Check | Result |
| --- | --- |
| Host GPU | RTX 5080, driver 580.178.04, 16303 MiB |
| Host PyTorch | 2.11.0+cu130, CUDA available |
| Host ONNX Runtime | 1.26.0, with CUDA and TensorRT providers listed |
| NVIDIA container toolkit | `nvidia-ctk`, `nvidia-container-runtime`, and `nvidia-container-cli` are absent. No CDI spec exists in `/etc/cdi` or `/var/run/cdi` |
| Docker GPU probe | `failed to discover GPU vendor from CDI: no known GPU vendor found` |
| Host `tritonserver` | Not installed |
| Triton image | `nvcr.io/nvidia/tritonserver:23.10-py3`, with ONNX Runtime backend 1.16.0 |
| Disk | 1.2 GB free on `/` (100% used) |
| Live inference backend | `.env` sets `INFERENCE_BACKEND=torch`. The live DINOv2 route runs local PyTorch, not Triton |

On 2026-10-09 a CPU-only Triton 23.10 probe ran with `models/` mounted read-only and explicit model control. Results:

- `dinov3_authenticity_candidate` version 1: `READY` on CPU.
- `dinov2_classifier` version 1: `UNAVAILABLE`, with `Unsupported model IR version: 10, max supported IR version: 9`. This export cannot load in Triton 23.10 on any device. Fixing GPU passthrough alone does not fix it.
- Server readiness was 400 because one requested model failed to load.

ONNX metadata:

- DINOv2: IR 10, opset 17, 572 external initializers, input `[batch, 3, 518, 518]`.
- DINOv3: IR 8, opset 18, no external data, input `[batch, 3, 512, 512]`.

### Security and integration findings

1. Research access is not role-restricted. `/research/verify` needs only an authenticated user plus `HYPEVAULT_DEPLOYMENT_MODE=research|shadow`. Roles are `buyer` and `seller`. `/auth/register` accepts the role from the request body. "Authorized research user" is not enforced.
2. About 105 GB of weights (124 `.pt`, `.onnx`, and `.onnx.data` files under `ml_rtx5080/experiments/`) are not gitignored. This includes `epoch_018.pt`. A broad `git add` would stage them.
3. `HYPEVAULT_DEPLOYMENT_MODE` is not set in `.env`, so the local stack resolves to `LEGACY_PRODUCTION`. In that state the research page always returns `POLICY_ERROR`. This is correct fail-closed behaviour, but it blocks any E2E run.
4. The DINOv2 research adapter calls `logits_to_verdict` only. The live route also applies `apply_min_authentic_confidence`. The same image can therefore get different DINOv2 verdicts on the two routes.
5. The DINOv2 research response sets `checkpoint_sha` to the placeholder `dinov2_classifier:1`, not an artifact SHA-256.
6. `infra/docker-compose.yml` mounts all of `models/` without explicit model control. Triton would try to load `dinov2_classifier` (which fails), `dinov3_authenticity_candidate`, and the duplicate `dinov3_authenticity_classifier` export. The compose Triton service also has no GPU reservation.
7. `/research/models` reports readiness to any authenticated user regardless of deployment mode. It returns no decisions. This is low risk.

### Model and data validity (separate from engineering)

- Data feasibility is `NO_GO`. There are 0 independently verified authentic and 0 verified counterfeit images. Every counterfeit file is a 512×512 JPEG sharing one quantization table. Labels come from directory names.
- DINOv3 checkpoint `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`, temperature `0.24038200410185356`, and threshold `0.50` are frozen.
- `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False`.
- Any engineering milestone below can reach READY while model validity stays unestablished. Serving parity proves only that two runtimes compute the same function. It does not prove that the function detects counterfeits.

## Milestones

### 1. GitHub release and merge readiness — IN_PROGRESS

Evidence: the branch is pushed. There is no pull request, no CI, and no branch protection. The clean checkout fails 1 of 14 tests, and only 2 of 48 test files are committed.

Deliverables:
- Ignore rules for weights and exports under `ml_rtx5080/experiments/`.
- A tracked, unmodified copy of the DINOv2 Triton config outside `models/`, so tests do not read gitignored paths.
- Commit the test files and small artifacts the safety suite needs, or make each test skip with an explicit reason when a local-only artifact is absent.
- A non-interactive ESLint config.
- A GitHub Actions workflow that runs the backend unit suite and the frontend typecheck, lint, and build.
- An opened pull request with the current blocker list in its description.

Dependencies: none outside the repository.

Acceptance criteria:
- A `git archive` extract of the branch passes the full committed unit suite. Any skips state their reason.
- `npm run lint` and `npm run build` run without prompts.
- `git status --porcelain --ignored` shows every `.pt`, `.onnx`, and `.onnx.data` file as ignored.
- CI passes on the pull request.
- Branch protection on `main` requires that CI and one review are recommended to the repository owner. Enabling it is the owner's decision.

Test commands:
```bash
rm -rf /tmp/hv_clean && mkdir /tmp/hv_clean && git archive HEAD | tar -x -C /tmp/hv_clean
cd /tmp/hv_clean && /path/to/.venv/bin/python -m unittest discover -s tests -v
cd frontend && npm ci && npm run lint && npx tsc --noEmit && npm run build
git status --porcelain --ignored ml_rtx5080/experiments | rg '\.(pt|onnx|onnx\.data)$'
gh pr view --json state,reviewDecision,statusCheckRollup
```

Known blockers: none technical. Merging needs an approval that does not exist yet.

Stop conditions:
- Any committed path contains weights, `.env`, datasets, or final-test images.
- A clean-checkout test fails.
- Merging without an approval.

Files likely to change: `.gitignore`, `tests/test_dual_model_router.py`, `tests/*`, `infra/triton/dinov2_classifier/config.pbtxt` (new tracked copy), `frontend/.eslintrc.json`, `.github/workflows/ci.yml`.

Production behaviour affected: no.

### 2. GPU runtime and dual-model Triton serving — BLOCKED

Evidence:
- No NVIDIA container toolkit and no CDI spec. The Docker GPU probe fails.
- The DINOv2 export is IR 10, and Triton 23.10 ONNX Runtime 1.16 supports at most IR 9.
- 1.2 GB of free disk. A newer Triton image is about 20 GB.

Sub-step READY: CPU Triton parity for `dinov3_authenticity_candidate`. It loads on CPU today.

Deliverables:
- DINOv3 Triton CPU parity against the FP32 PyTorch reference, using tolerances pre-registered in `model_router_contract.json`: logit `1e-4`, probability `1e-5`. Inputs are eligible non-final-test images, at batch sizes 1, 2, 4, and 8.
- A written decision on the DINOv2 IR problem. Option A: a newer Triton release whose ONNX Runtime supports IR 10. Option B: a new `dinov2_classifier` version 2 exported at IR ≤ 9, with version 1 left untouched and parity against version 1 proven.
- A Triton image that supports Blackwell (sm_120), run through the NVIDIA container toolkit. Installing the toolkit is a system change the owner must approve.
- Explicit model control in compose, loading only `dinov2_classifier` and `dinov3_authenticity_candidate`.
- Per-model latency and memory, both sequential and concurrent.

Dependencies:
- Owner approval to install `nvidia-container-toolkit`, without replacing the driver.
- About 25 GB of free disk.
- A Triton release that supports sm_120 and ONNX IR 10, or the Option B re-export.

Acceptance criteria:
- `docker run --gpus all <triton image> nvidia-smi` lists the RTX 5080.
- `/v2/health/ready` returns 200, and each model's `/ready` returns 200 at version 1 (or the agreed DINOv2 version).
- DINOv3 parity is within the pre-registered tolerances, with 0 decision mismatches at 0.50. If parity fails, tolerances are not widened. The candidate is not presented as an equivalent.
- GPU use is shown in Triton metrics (`nv_gpu_utilization`, `nv_gpu_memory_used_bytes`) during inference before GPU serving is claimed.
- A missing model returns `MODEL_ERROR` with no decision against the live server.

Test commands:
```bash
docker run --rm --gpus all <triton-image> nvidia-smi
docker run -d --rm --name hv-triton --gpus all -v "$PWD/models:/models:ro" -p 18000:8000 -p 18001:8001 -p 18002:8002 \
  <triton-image> tritonserver --model-repository=/models --model-control-mode=explicit \
  --load-model=dinov2_classifier --load-model=dinov3_authenticity_candidate
curl -s localhost:18000/v2/health/ready -o /dev/null -w '%{http_code}\n'
curl -s -X POST localhost:18000/v2/repository/index
curl -s localhost:18002/metrics | rg 'nv_gpu_(utilization|memory_used_bytes)'
TRITON_PORT=18001 .venv/bin/python -m unittest tests.test_backend_parity_v2 tests.test_gpu_serving -v
```

Known blockers: the toolkit is absent, the DINOv2 IR 10 export is incompatible with Triton 23.10, disk is full, and Triton 23.10 predates Blackwell.

Stop conditions:
- Any step requires replacing the NVIDIA driver.
- Parity fails at the pre-registered tolerance.
- `dinov2_classifier` version 1 would be overwritten.
- Freeing disk would delete checkpoints or Phase 45–49 artifacts.

Files likely to change: `infra/docker-compose.yml` (Triton service only), `infra/triton/*/config.pbtxt`, local `models/` (gitignored), `ml_rtx5080/experiments/dual_model_triton_v1/parity_results.json`, `ml_rtx5080/experiments/dual_model_triton_v1/latency_and_memory_results.json`.

Production behaviour affected: only if the live route moves from `INFERENCE_BACKEND=torch` to Triton or DINOv2 is re-exported. Either change requires DINOv2 parity first.

### 3. Research UI and end-to-end application tests — IN_PROGRESS

Evidence: the selector, the experimental notice, the allowlist, and the live-route rejection are committed and unit-tested with mocks. A browser check covered selection and the notice only. The role gap, the DINOv2 verdict divergence, and the placeholder SHA are open. Live inference E2E needs Milestone 2.

Deliverables:
- A server-assigned research role or allowlist on `/research/verify` and `/research/models`. Self-registration must not grant it.
- DINOv2 research verdicts computed with the same policy as the live route, including the minimum-authentic-confidence rule.
- A real DINOv2 artifact SHA-256, computed once at startup and not per request.
- A documented local research launch that sets `HYPEVAULT_DEPLOYMENT_MODE=research` outside `infra/docker-compose.yml` (the customer-boundary test forbids it there).
- Browser E2E for all 12 Phase 50 flows.

Dependencies: Milestone 1 merged. Milestone 2 for live-inference flows. PostgreSQL for authenticated flows.

Acceptance criteria:
- A buyer or seller account gets 403 on research endpoints.
- Selecting DINOv2 and DINOv3 returns `LEGACY_DINOV2` and `DINOV3_RESEARCH_PROTOTYPE` respectively from a live Triton call.
- Unsupported brand, missing brand, invalid image, production-mode DINOv3, and a stopped model each return a non-verdict status.
- No listing row is created by any research request.
- Legacy listing labels are unchanged.

Test commands:
```bash
.venv/bin/python -m unittest tests.test_dual_model_router tests.test_customer_boundary tests.test_research_demo_hardening -v
docker compose -f infra/docker-compose.yml up -d postgres redis
cd backend && HYPEVAULT_DEPLOYMENT_MODE=research ../.venv/bin/uvicorn main:app --port 8000
cd frontend && npm run build && npm run start
psql "$DATABASE_URL" -c 'select count(*) from listings'   # before and after research requests
```

Known blockers: live-inference flows depend on Milestone 2. The role model does not exist yet.

Stop conditions:
- Any change to `AUTHENTICITY_MODEL_PRODUCTION_APPROVED`.
- DINOv3 becomes reachable from `/verify/authenticate` or the seller upload page.
- A research request writes a listing.

Files likely to change: `backend/auth/deps.py`, `backend/database.py` (role enum, which needs a migration that is not run in this roadmap step), `backend/inference/research_routes.py`, `backend/inference/model_router.py`, `frontend/src/app/research/page.tsx`, `tests/test_dual_model_router.py`, new E2E tests.

Production behaviour affected: the role change touches authentication for every user and needs a migration. Research inference itself does not affect production.

### 4. Independently verified data acquisition and provenance — BLOCKED

Evidence: the Phase 49 decision is `NO_GO`. 0 authentic and 0 counterfeit images are verified, there are 0 independent source groups, and 16 candidate sources were rejected.

Deliverables:
- Signed agreements with a qualified examiner or owner-permission photography source.
- Images admitted under `image_label_evidence_v1`, with rights recorded separately from label evidence.
- A shared capture and export geometry policy (`shared_original_plus_resize_pad_square_v1`), so class and file format are not confounded.

Dependencies: an external partner. None is committed today.

Acceptance criteria:
- The Phase 49 pilot target (30 authentic and 30 counterfeit across 3 brands) is met with verifier identity, method, and licence recorded per image.
- Counterfeit and authentic files share export conventions, checked by the Phase 47/48 census scripts.
- The manifest hash is recorded before any model sees the data.

Test commands:
```bash
.venv/bin/python -m unittest tests.test_verified_data_pilot tests.test_dataset_rebuild tests.test_dataset_integrity -v
```

Known blockers: no verifier, no licence, no source.

Stop conditions:
- Any label is inferred from a directory name, seller claim, or model output.
- Rights are unclear.

Files likely to change: `ml_rtx5080/experiments/verified_data_pilot_v2/` (new), and the admission manifest.

Production behaviour affected: no.

### 5. Cross-brand research and model improvement — BLOCKED

Evidence: depends on Milestone 4. The current catalog is quarantined from training.

Deliverables:
- A pre-registered protocol with leave-one-brand-out folds on verified data only.
- A held-out set that no model or threshold search has seen.

Dependencies: Milestone 4 accepted.

Acceptance criteria:
- The protocol hash is recorded before training.
- Results are reported against the frozen DINOv3 baseline without changing its checkpoint, temperature, or threshold.

Test commands: the protocol tests added with the protocol, plus the full unit suite.

Known blockers: no verified data.

Stop conditions:
- Training on the current catalog.
- Reuse of the Phase 45 final test.
- Any edit to the frozen constants.

Files likely to change: new `ml_rtx5080/experiments/*_v3/` directories and a new training config. No backend changes.

Production behaviour affected: no.

### 6. Final evaluation and production approval decision — BLOCKED

Evidence: depends on Milestones 2, 4, and 5. The production flag is false.

Deliverables:
- A single final evaluation on an untouched verified holdout.
- Out-of-distribution and unseen-brand behaviour.
- A serving-parity record from Milestone 2.
- A written approval decision by an accountable owner.

Dependencies: Milestones 2, 4, and 5 accepted. A named approver.

Acceptance criteria:
- Pre-registered metrics, including the false-authentic rate, meet thresholds set before the evaluation.
- The approval is recorded in the repository before `AUTHENTICITY_MODEL_PRODUCTION_APPROVED` changes.

Test commands: the full unit suite and the evaluation protocol tests.

Known blockers: everything upstream.

Stop conditions:
- The holdout is used more than once.
- Thresholds are tuned on the holdout.
- No approver is named.

Files likely to change: `backend/inference/publication_gate.py` (only after a recorded approval), and evaluation artifacts.

Production behaviour affected: yes. This is the only milestone that may change it.

## Recommended first task

Make `feature/dual-model-triton` self-verifying from a clean checkout and open the pull request (Milestone 1).

Acceptance criteria:

1. `.gitignore` excludes `*.pt`, `*.onnx`, and `*.onnx.data` under `ml_rtx5080/experiments/`. `git status --porcelain --ignored` shows `epoch_018.pt` as ignored.
2. The DINOv2 Triton config is tracked under `infra/triton/dinov2_classifier/config.pbtxt`, byte-identical to `models/dinov2_classifier/config.pbtxt`. The test reads the tracked copy.
3. A `git archive` extract passes the committed unit suite with 0 failures. Every skip names its missing dependency.
4. `npm run lint` runs non-interactively and passes. `npx tsc --noEmit` and `npm run build` pass.
5. A CI workflow runs steps 3 and 4 on the pull request and passes.
6. A pull request from `feature/dual-model-triton` to `main` is open. Its description lists the Milestone 2 blockers and the security findings above. It is not merged.
7. No application behaviour, model artifact, calibration value, threshold, or final-test file changes.

The second task is DINOv3 Triton CPU parity. It is the only serving check that can run today without system changes.
