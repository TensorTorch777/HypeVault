# HypeVault

AI-gated marketplace for authentic sneakers and ultra-luxury watches — **buy-now style flows only** (no auctions, no bidding language, no WebSockets).

## Superpowers (agent skills)

The npm package `@obra/superpowers` is **not published**; Superpowers is distributed as a [Cursor/Claude plugin](https://github.com/obra/superpowers) and skill library.

- **Cursor:** install `superpowers` from the plugin marketplace (or run `/add-plugin superpowers` in Agent chat).
- **Offline reference:** `./scripts/bootstrap_superpowers.sh` clones upstream into `vendor/superpowers`.

Domain skills for this repo live in `.cursor/skills/` (`authentication.md`, `dinov2_inference.md`, `playwright_scraper.md`, `price_dashboard.md`, `image_upload.md`, `aws_infra.md`).

## Stack

- **Backend:** Python 3.11+, FastAPI (async), SQLAlchemy 2 + asyncpg, Redis (`redis-py` asyncio; Python 3.12+ replaces legacy `aioredis`), boto3/aioboto3, Playwright, Triton gRPC client, Prometheus `/metrics`.
- **Frontend:** Next.js 14 (App Router), TypeScript, Tailwind, TanStack Query, Axios, Framer Motion, Recharts.
- **Infra:** Docker Compose (Postgres, Redis, API, optional Triton), ECS/SQS helpers under `infra/`.

## Quick start (local)

**Fast path (Docker + venv + migrate + seed):** from repo root, with Docker running:

```bash
bash scripts/dev_setup.sh
```

This creates `.venv`, installs Python deps, starts **Postgres + Redis** via Compose, runs **Alembic**, and runs **`scripts/seed_database.py`**. A root **`.env`** is used (see `.env.example`); `backend/config.py` loads it from the repo root.

1. **Python environment** (if you skipped `dev_setup.sh`)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. **Database** (if you skipped `dev_setup.sh`)

```bash
# Start Postgres + Redis (requires Docker; use sudo if your user lacks docker group access)
docker compose -f infra/docker-compose.yml up -d postgres redis

cd backend
cp ../.env.example ../.env   # or keep the generated .env from dev_setup
alembic upgrade head
cd ..
python3 scripts/seed_database.py
```

3. **API**

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. **Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. API: `http://localhost:8000`. Triton gRPC from host (Compose): `localhost:18001` (see `.env.example`).

## Key API routes

- `POST /auth/register`, `POST /auth/login`, `POST /auth/google`, `POST /auth/refresh`, `POST /auth/logout`
- `POST /verify/authenticate` (multipart image + optional `listing_id`)
- `POST /listings/`, `GET /listings/`, `GET /listings/recent`, `GET /listings/compare?q=...`, `GET /listings/{id}/comparison`
- `GET /metrics`, `GET /health`, `GET /health/ready` (Postgres + Redis)

## Notes

- **S3:** With empty AWS credentials, uploads are stored under `backend/local_uploads` and served at `GET /static/local/...`.
- **Redis:** Required for auth refresh rotation and scraper caching.
- **Triton:** Model name `dinov2_classifier`, inputs `input__0` shape `[1,3,518,518]` FP32 — see `scripts/export_tensorrt.py` / `scripts/setup_triton.sh`.
- **Local trained model (no Triton):** `pip install -r requirements_inference.txt`, copy your checkpoint to e.g. `models/hypevault_classifier.pt`, then set `INFERENCE_BACKEND=torch`, `LOCAL_MODEL_PATH=models/hypevault_classifier.pt`, and `DINOV2_MODEL_NAME=dinov2_vitg14_reg` (must match training). Switch back to `INFERENCE_BACKEND=triton` when the server is ready.

## Production expectations

- Never commit secrets; use environment variables only.
- Front-end uploads should prefer **S3 pre-signed PUT** via `POST /listings/presign`.
- External scrapers are best-effort; DOMs change — failures return empty platform rows and JSON errors (not opaque 500s).
