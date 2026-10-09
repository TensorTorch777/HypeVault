# HypeVault

<p align="center">
  <strong>Five-brand authenticity research prototype</strong><br/>
  Not a universal luxury-watch authenticator, and not ready for production promotion.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-111111?style=for-the-badge"></a>
  <img alt="Frontend" src="https://img.shields.io/badge/frontend-Next.js%2014-000000?style=for-the-badge">
  <img alt="Backend" src="https://img.shields.io/badge/backend-FastAPI-0A0A0A?style=for-the-badge">
  <img alt="Inference" src="https://img.shields.io/badge/inference-Triton%20%7C%20Torch-222222?style=for-the-badge">
  <img alt="Database" src="https://img.shields.io/badge/database-PostgreSQL%20%7C%20Redis-1B1B1B?style=for-the-badge">
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-5B21B6?style=flat-square">
  <img alt="API" src="https://img.shields.io/badge/api-FastAPI%20Async-0E7490?style=flat-square">
  <img alt="UI" src="https://img.shields.io/badge/ui-Tailwind%20%2B%20Motion-7C3AED?style=flat-square">
  <img alt="Auth" src="https://img.shields.io/badge/auth-JWT%20%2B%20OAuth-334155?style=flat-square">
  <img alt="Observability" src="https://img.shields.io/badge/metrics-Prometheus-7F1D1D?style=flat-square">
</p>

---

## Quick Navigation

- [Overview](#overview)
- [Core Capabilities](#core-capabilities)
- [Architecture Snapshot](#architecture-snapshot)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Command Center by Role](#command-center-by-role)
- [API Surface (Key Routes)](#api-surface-key-routes)
- [Demo Script (Presentation Flow)](#demo-script-presentation-flow)
- [Inference Modes](#inference-modes)
- [Model training results](#model-training-results)
- [Security and Secret Handling](#security-and-secret-handling)
- [Production Notes](#production-notes)
- [License](#license)

---

## Current status

`NOT_READY_FOR_PROMOTION`

## Scope

Five-brand authenticity research prototype.

Supported brands:

1. A. Lange & Söhne
2. Audemars Piguet
3. Patek Philippe
4. Richard Mille
5. Vacheron Constantin

General luxury-watch authenticity is unsupported. Open-set rejection is not validated. `production_ood_threshold` is null.

## Validation summary

- Final test: 3,006 in-distribution samples, five brands, calibrated ROC-AUC 1.0, PR-AUC 1.0, F1 1.0, false-authentic 0, false-fake 0. This is not universal authenticity evidence.
- Calibration: 2,998 samples, frozen temperature `0.24038200410185356`.
- Robustness: clean false-authentic 0 and false-fake 0. Quality policy caught 8 of 9 known stress false-fake events. Resize/recompression remains uncovered.
- OOD: Phase 35 is `OOD_FAIL` (337/408 authentic escape; unseen luxury about 88.1%). Phase 36 is `WATCH_OOD_UNRESOLVED` (25 holdout images).
- Serving: live listing checks stay on DINOv2 (`dinov2_vitb14_live`, ViT-B/14 at 504). The frozen research candidate is DINOv3 (`dinov3_authenticity_candidate`). Both are served together on NVIDIA Triton 26.01 GPU with exact FP32 (`use_tf32=0`). The old ViT-G/14 `dinov2_classifier` is not routed. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED` remains false.

## Known blockers

- OOD fail
- Runtime parity between the live DINOv2 path and the frozen DINOv3 candidate
- Serving infrastructure
- Policy approval

## Overview

HypeVault is a marketplace codebase aimed at luxury-watch listings.
The authenticity model is a research prototype for the five brands above.
A declared brand outside that list returns `UNSUPPORTED_SCOPE` and no authentic or fake verdict.

Core principles:
- No authenticity verdict outside the five-brand scope
- Comparable market context is separate from authenticity
- Customer-facing production promotion is forbidden

---

## Core Capabilities

- AI-backed listing verification with Triton or local Torch fallback
- Role-based buyer/seller experiences with JWT session flows
- Google sign-in support (buyer path) plus email/password auth
- Multi-source market comparison (StockX, Chrono24, eBay)
- Cache freshness metadata for comparison responses
- Health, readiness, and metrics endpoints for runtime visibility

---

## Architecture Snapshot

```mermaid
flowchart LR
  U[Buyer / Seller] --> F[Next.js Frontend]
  F --> B[FastAPI Backend]
  B --> PG[(PostgreSQL)]
  B --> R[(Redis)]
  B --> T[Triton Inference]
  B --> S3[(S3 or Local Upload Storage)]
  B --> SC[Scraper Workers]
  SC --> X[StockX]
  SC --> C[Chrono24]
  SC --> E[eBay]
```

### Frontend
- Next.js 14 App Router
- TypeScript + Tailwind CSS
- TanStack Query + Axios
- Framer Motion + Recharts

### Backend
- FastAPI (async)
- SQLAlchemy 2 + asyncpg
- Redis (auth token rotation + scraper cache)
- Playwright scrapers
- Triton gRPC inference client
- Prometheus metrics endpoint

### Infrastructure
- Docker Compose for local Postgres/Redis/Triton
- ECS/SQS helper assets under `infra/`
- S3 upload flow with local fallback

---

## Repository Structure

```text
backend/      FastAPI app, auth, listings, inference, scraper
frontend/     Next.js UI and client integrations
infra/        Docker and deployment helper artifacts
ml/           Training and model utility scripts (DINOv2-Giant; RTX 6000 Pro Blackwell reference run)
ml_rtx5080/   ViT-B / 504px training pipeline and checkpoints (RTX 5080)
scripts/      Python utilities; optional local *.sh helpers are gitignored
```

---

## Quick Start

### 1) Fast path

From repo root (with Docker running). Shell helpers under `scripts/` are **not** tracked in git; use the manual path below or keep your own local `dev_setup.sh`.

### 2) Manual path (recommended clone)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
docker compose -f infra/docker-compose.yml up -d postgres redis
cd backend && alembic upgrade head && cd ..
python3 scripts/seed_database.py
```

Start API:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Start frontend:

```bash
cd frontend
npm install
npm run dev
```

Application URLs:
- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- Triton gRPC (host): `localhost:18001`

Environment template:
- Copy `.env.example` to `.env` and fill runtime credentials before non-local deployment.

---

## Command Center by Role

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>Developer</h3>
      <p>Run application stack for feature work.</p>
      <pre><code>cd ~/Desktop/HypeVault
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
docker compose -f infra/docker-compose.yml up -d postgres redis
cd backend && alembic upgrade head && cd .. && python scripts/seed_database.py
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
# new terminal
cd frontend && npm install && npm run dev</code></pre>
    </td>
    <td width="33%" valign="top">
      <h3>Infrastructure</h3>
      <p>Bring dependencies and readiness online.</p>
      <pre><code>cd ~/Desktop/HypeVault
docker compose -f infra/docker-compose.yml up -d postgres redis triton
curl -sS http://localhost:8000/health
curl -sS -i http://localhost:8000/health/ready</code></pre>
    </td>
    <td width="33%" valign="top">
      <h3>ML / Inference</h3>
      <p>Export and run model path.</p>
      <pre><code>cd ~/Desktop/HypeVault
. .venv/bin/activate
python scripts/export_tensorrt.py
# fallback mode
export INFERENCE_BACKEND=torch
export LOCAL_MODEL_PATH=models/hypevault_classifier.pt</code></pre>
    </td>
  </tr>
</table>

---

## API Surface (Key Routes)

<details open>
  <summary><strong>Authentication</strong></summary>

- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/google`
- `POST /auth/refresh`
- `POST /auth/logout`
- `GET /auth/me`

</details>

<details open>
  <summary><strong>Verification and Listings</strong></summary>

- `POST /verify/authenticate` — legacy DINOv2 listing check. `model = LEGACY_DINOV2`, `research_candidate = false`, `production_validation = NOT_ESTABLISHED`. It is not the DINOv3 research prototype. The brand field is user-declared. Unsupported brands return HTTP 422 `UNSUPPORTED_SCOPE` with `decision = null`.
- `POST /research/verify` — frozen DINOv3 five-brand research prototype. Requires `HYPEVAULT_DEPLOYMENT_MODE=research` or `shadow`. Production, missing, and unknown modes return `POLICY_ERROR` with `decision = null`. `brand_verification = NOT_PERFORMED`, `research_only = true`, `production_ready = false`.
- `POST /listings/`
- `GET /listings/`
- `GET /listings/recent`
- `GET /listings/compare?q=...`
- `GET /listings/{id}/comparison`
- `POST /listings/presign`

</details>

<details>
  <summary><strong>Operations</strong></summary>

- `GET /health`
- `GET /health/ready`
- `GET /metrics`

</details>

---

## Demo Script (Presentation Flow)

Use this minimal runbook for a polished live demo:

1. **Open landing page** and position the narrative: AI-gated trust layer for luxury commerce.
2. **Authenticate as seller**, create/upload listing, run verification.
3. **Show verdict + confidence** and explain gating logic.
4. **Open comparison panel** and highlight cross-market pricing context.
5. **Switch to buyer perspective** and show curated discovery flow.
6. **Close with operations proof** using `GET /health/ready` and `GET /metrics`.

Presentation-ready command block:

```bash
curl -sS http://localhost:8000/health
curl -sS -i http://localhost:8000/health/ready
curl -sS http://localhost:8000/metrics | sed -n '1,20p'
```

---

## Inference Modes

Two allowlisted Triton models, selected by a logical id. Triton model names are server-side and are rejected as client selectors.

| Logical id | Triton model | Artifact | Input | Role |
| --- | --- | --- | --- | --- |
| `dinov2_legacy` | `dinov2_vitb14_live` v1 | live ViT-B/14 checkpoint `fe1daa0b…fa66` | `[1,3,504,504]` FP32 | live listing check and research selector |
| `dinov3_experimental` | `dinov3_authenticity_candidate` v1 | frozen `epoch_018.pt` `5a38c93f…d28f`, temperature `0.24038200410185356` | `[1,3,512,512]` FP32 | research/shadow only |

Validated GPU serving is `KIND_GPU` with `use_tf32=0`. The SHA-256 of the DINOv3 checkpoint is verified at API startup and cached; a mismatch fails closed (`decision = null`) and is not hashed again on every request. Production mode cannot select DINOv3. Neither model auto-publishes a listing.

The unrouted `dinov2_classifier` ViT-G/14 518 artifact is not a substitute for either selector model.

### Local Torch fallback
Use when Triton is unavailable for the live DINOv2 listing path:

```bash
pip install -r requirements_inference.txt
```

Then configure in `.env`:
- `INFERENCE_BACKEND=torch`
- `LOCAL_MODEL_PATH=models/hypevault_classifier.pt`
- `DINOV2_MODEL_NAME=dinov2_vitg14_reg`

Switch back to `INFERENCE_BACKEND=triton` for production parity.

---

## Model training results

Binary authenticity classifier: **Authentic (label 0)** vs **Deepfake (label 1)** — see dataset layout in `ml/train.py`.  
Below: **training curves** and **validation confusion-matrix dashboards** for two training setups.

### NVIDIA RTX 6000 Pro Blackwell (`ml/`)

Full fine-tune pipeline (**DINOv2-Giant**, Stage 2). Validation eval figure summarizes ~**6,000** held-out images (Authentic vs Deepfake).

<p align="center">
  <img src="ml/checkpoints/training_curves.png" alt="HypeVault training curves — DINOv2-Giant on RTX 6000 Pro Blackwell (ml/checkpoints)" width="780">
</p>

<p align="center">
  <img src="ml/checkpoints/confusion_matrix_eval.png" alt="HypeVault confusion matrix and eval metrics — RTX 6000 Pro Blackwell (ml/checkpoints)" width="780">
</p>

### NVIDIA GeForce RTX 5080 (`ml_rtx5080/`)

Smaller backbone / resolution run (see `ml_rtx5080/train.py`).

<p align="center">
  <img src="ml_rtx5080/checkpoints/training_curves.png" alt="HypeVault training curves — RTX 5080 (ml_rtx5080/checkpoints)" width="780">
</p>

<p align="center">
  <img src="ml_rtx5080/checkpoints/confusion_matrix_eval.png" alt="HypeVault confusion matrix eval — RTX 5080 (ml_rtx5080/checkpoints)" width="780">
</p>

> Checkpoint weights (`.pt`, `.onnx`, etc.) stay **gitignored**; only these PNG artifacts are tracked for documentation.

---

## Security and Secret Handling

- Never commit secrets.
- Keep runtime values in environment variables only.
- Use a local env template (keep secrets out of git).
- `.gitignore` is configured to exclude:
  - `.env*` and `.env.example`
  - key/cert artifacts
  - service-account JSON files
  - heavyweight training and dataset folders

Before each push:
- verify no secret-bearing files are staged
- keep machine-specific `.env` and credential material local-only

---

## Production Notes

Customer-facing promotion of the authenticity model is forbidden. The notes below are operational only.

- Prefer S3 pre-signed uploads via `POST /listings/presign`
- Ensure Redis is healthy for token rotation and cache paths
- Treat external scraping as best-effort and failure-tolerant
- Monitor readiness and metrics before exposing user traffic
- Keep Triton model readiness green before enabling verification-dependent flows

---

## License

MIT License. See `LICENSE`.
