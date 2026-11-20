# HypeVault — AWS deployment and verification report

**Contributor:** Harsh Kankariya  
**Date:** 21 November 2026 (IST)  
**Working branch:** `deployment/aws-hypevault` (no changes to `main`)  
**Purpose:** Deploy the team's HypeVault final-year project for a cloud-hosted presentation.

## Deployment architecture

| Layer | Technology | EC2 deployment |
| --- | --- | --- |
| Compute | Amazon EC2, Ubuntu 24.04, `t3.large` | Running in `ap-southeast-2` (Sydney) |
| Web UI | Next.js 14 | Docker; public HTTP port 80 |
| API | FastAPI / Uvicorn | Docker; internal port 8000 (not exposed publicly) |
| Database | PostgreSQL 15 | Docker; private service and persistent volume |
| Cache | Redis 7 | Docker; private service and persistent volume |
| Runtime orchestration | Docker Compose | `infra/docker-compose.aws.yml` |
| Persistent uploads | Docker named volume | Preserved between container rebuilds |
| Configuration | `.env.aws` on EC2 | Random generated secrets, not committed |

The app is currently available at `http://3.106.136.202/`. This is the instance's
**current, non-static public address**; a stop/start can change it.

The deployment uses purpose-specific production-style Dockerfiles and a separate
Compose file. This avoids the repository's development-mode `--reload` server,
bind-mounted node modules, known development credentials, and public PostgreSQL/
Redis ports. Next.js routes server-side API requests to the `backend` Compose
service; browsers use the public web server rather than `localhost:8000`.

## Observed verification results

The following checks were observed on the deployed EC2 host or in an external
browser. These are **deployment results**, not proof of working GPU inference.

| Test | Result | Evidence / limitation |
| --- | --- | --- |
| `docker compose ... ps` | **PASS** | Backend and frontend running; Postgres and Redis healthy |
| Alembic migrations | **PASS** | Database upgrades through `20261017_0004` completed |
| `GET /health` via public web port | **PASS** | HTTP 200, `{"status":"ok"}` |
| Homepage | **PASS** | Public luxury-watch landing page rendered |
| Benchmark | **PASS** | `/benchmark` displays existing benchmark documentation |
| Method | **PASS** | `/#method` shows three fixed methodology cards; `/method` is not a route |
| Market UI | **PARTIAL** | Page/search renders, but price feeds are not yet populated |
| Research | **BLOCKED** | Requires SSO; Google OAuth client IDs not yet configured |
| `GET /health/ready` | **DEGRADED** | `postgres=ok`, `redis=ok`, `triton=error`, `market_ebay=not_configured`, `dinov3_checkpoint=error` |

## Unfinished integration items

1. **GPU / AI model:** `t3.large` has no GPU. Phase 76 DINOv3/Perception Encoder
   screening needs CUDA and trusted model artifacts. The AWS quota request for
   **4 G/VT-family vCPUs** was opened and was awaiting approval at verification
   time. The local `epoch_018.pt` SHA-256 matched the project's expected digest.
2. **Triton:** The existing repository has model configurations, but production
   inference additionally requires a GPU-enabled host, NVIDIA runtime, valid
   ONNX model exports, a Triton model repository, and parity tests. The legacy
   Compose Triton service alone does not enable Phase 76.
3. **Market data:** eBay Browse API credentials are absent. The importer
   `scripts/import_chrono24_market.py` requires a dated, appropriately sourced
   CSV snapshot. Dealer crawler code exists but has not run as part of this EC2
   deployment; dealer terms and access permissions must be checked before use.
4. **Authentication / HTTPS:** The public deployment currently uses HTTP.
   Google SSO is not configured; password login is intentionally disabled.
   Do not enter real credentials or claim production authentication readiness.

## Useful operational commands

Run in the EC2 Instance Connect terminal from `~/HypeVault`:

```bash
sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml ps
sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml logs --tail=80 backend frontend
curl -i http://localhost/health
curl -sS http://localhost/health/ready
```

To restart the services without rebuilding:

```bash
sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml up -d
```

Do not commit `.env.aws`, AWS private keys, model weights, downloaded datasets,
or authentication tokens. Do not open database/cache/model-service ports to the
public internet. To control AWS costs, **stop the EC2 instance after the
presentation**, once it is no longer needed; note that EBS storage charges can
continue, and the public IPv4 address can change after a stop/start. Stopping
Docker containers alone does not stop EC2 instance charges.

## Contributor summary for the final-year report

Configured and deployed the HypeVault full-stack web application on AWS EC2,
containerized the Next.js frontend and FastAPI backend, provisioned persistent
PostgreSQL and Redis services through Docker Compose, managed environment
secrets separately from Git, initialized the database schema, verified public
HTTP and API health, and documented limitations and the GPU/Triton upgrade
path. All deployment work was committed on the dedicated
`deployment/aws-hypevault` branch rather than `main`.
