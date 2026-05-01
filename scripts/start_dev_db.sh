#!/usr/bin/env bash
# Start Postgres + Redis (Docker), wait for DB, migrate, seed.
# Run from repo root:  bash scripts/start_dev_db.sh
# If you see "permission denied" on docker.sock, either:
#   sudo bash scripts/start_dev_db.sh
# or (one-time fix):  sudo usermod -aG docker "$USER"  then log out and back in.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pick_docker() {
  if docker info >/dev/null 2>&1; then
    echo docker
  elif sudo -n docker info >/dev/null 2>&1; then
    echo "sudo -n docker"
  else
    echo sudo docker
  fi
}

DOCKER="$(pick_docker)"
COMPOSE="$DOCKER compose -f $ROOT/infra/docker-compose.yml"

echo "==> Starting Postgres + Redis ($DOCKER)..."
$COMPOSE up -d postgres redis

echo "==> Waiting for Postgres (up to 45s)..."
for i in $(seq 1 45); do
  if $COMPOSE exec -T postgres pg_isready -U hypevault -d hypevault >/dev/null 2>&1; then
    echo "    Postgres is ready."
    break
  fi
  sleep 1
  if [[ "$i" -eq 45 ]]; then
    echo "    Timed out. Check: $DOCKER compose -f $ROOT/infra/docker-compose.yml logs postgres" >&2
    exit 1
  fi
done

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "==> Creating .venv"
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install -q -r "$ROOT/requirements.txt"

echo "==> Alembic migrate"
(cd "$ROOT/backend" && alembic upgrade head)

echo "==> Seed demo users + listings"
python3 "$ROOT/scripts/seed_database.py"

echo ""
echo "Done. Start API:  cd backend && source ../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo "Login: seller@hypevault.demo / Seller123"
