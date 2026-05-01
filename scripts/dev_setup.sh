#!/usr/bin/env bash
# One-shot local setup: Docker Postgres/Redis, Python venv, migrations, seed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Starting Postgres + Redis (Docker)"
docker compose -f infra/docker-compose.yml up -d postgres redis

echo "==> Waiting for Postgres..."
for i in $(seq 1 30); do
  if docker compose -f infra/docker-compose.yml exec -T postgres pg_isready -U hypevault -d hypevault >/dev/null 2>&1; then
    echo "    Postgres is ready."
    break
  fi
  sleep 1
  if [[ "$i" -eq 30 ]]; then
    echo "    Timed out waiting for Postgres." >&2
    exit 1
  fi
done

if [[ ! -d .venv ]]; then
  echo "==> Creating Python venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> pip install"
pip install -q -r requirements.txt

echo "==> Alembic migrate"
cd backend
alembic upgrade head
cd "$ROOT"

echo "==> Seed demo users + listings"
python3 scripts/seed_database.py

echo ""
echo "Done. Next steps:"
echo "  API:    cd backend && source ../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo "  Check:  curl -s http://localhost:8000/health/ready  (expect postgres+redis ok)"
echo "  Web:    cd frontend && npm install && npm run dev   (or npm run dev:3001 — CORS includes both ports)"
echo "  Login:  seller@hypevault.demo / Seller123   |   buyer@hypevault.demo / Buyer123"
echo "  Env:    copy .env.example → .env ; set NEXT_PUBLIC_API_URL if API is not :8000"
