#!/usr/bin/env bash
# One-shot local stack: Postgres + Redis (Docker), migrations, seed, then prints commands for API + Next.js.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV_PY="${ROOT}/.venv/bin/python"
VENV_UVICORN="${ROOT}/.venv/bin/uvicorn"
COMPOSE=(docker compose -f "${ROOT}/infra/docker-compose.yml")

echo "== HypeVault local stack (repo root: ${ROOT})"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "== Starting Postgres + Redis"
  "${COMPOSE[@]}" up -d postgres redis
  echo "== Waiting for database (15s max)"
  for _ in $(seq 1 15); do
    if "${COMPOSE[@]}" exec -T postgres pg_isready -U hypevault -d hypevault >/dev/null 2>&1; then
      echo "   Postgres is up."
      break
    fi
    sleep 1
  done
else
  echo "!! Docker not available or daemon not running."
  echo "   Ensure Postgres on localhost:5432 (db=hypevault user/pass=hypevault) and Redis on :6379,"
  echo "   or fix Docker permissions and re-run this script."
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "!! No .venv at ${ROOT}/.venv — create it and pip install -r requirements.txt backend deps."
  exit 1
fi

if command -v nc >/dev/null 2>&1 && nc -z 127.0.0.1 5432 2>/dev/null; then
  echo "== Alembic migrations"
  (cd "${ROOT}/backend" && "$VENV_PY" -m alembic upgrade head)
  echo "== Seed demo users + listings (idempotent)"
  DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://hypevault:hypevault@localhost:5432/hypevault}" \
    REDIS_URL="${REDIS_URL:-redis://localhost:6379}" \
    "$VENV_PY" "${ROOT}/scripts/seed_database.py"
else
  echo "!! Postgres not reachable on 127.0.0.1:5432 — skipping migrations and seed."
  echo "   After DB is up:  cd backend && ../.venv/bin/python -m alembic upgrade head"
  echo "                    ../.venv/bin/python scripts/seed_database.py"
fi

echo ""
echo "== Stack ready. Run these in separate terminals (repo .env is loaded automatically by FastAPI):"
echo ""
echo "  Terminal A — API + AI inference:"
echo "    cd ${ROOT}/backend && ${VENV_UVICORN} main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  Terminal B — Next.js (uses frontend/.env.local for NEXT_PUBLIC_API_URL):"
echo "    cd ${ROOT}/frontend && npm run dev"
echo ""
echo "  Demo login (after seed): seller@hypevault.demo / Seller123  |  buyer@hypevault.demo / Buyer123"
echo "  Open http://localhost:3000 — API http://localhost:8000  |  health: GET /health"
echo ""
