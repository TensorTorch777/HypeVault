# HypeVault presentation deployment (AWS EC2)

This is a **web/API preview**, not a GPU-inference deployment. It leaves Triton
offline and authentication disabled until HTTPS + Google OAuth are configured.
Model-screening endpoints need a GPU, full trusted model artifacts, and testing.

## Safe branch-only Git changes (Windows PowerShell)

1. Copy `infra/Dockerfile.backend.aws`, `infra/Dockerfile.frontend.aws`,
   `infra/docker-compose.aws.yml`, and this guide into your existing
   `C:\Users\Dell\Downloads\HypeVault-Git\infra` folder.
2. In PowerShell:

   ```powershell
   cd C:\Users\Dell\Downloads\HypeVault-Git
   git branch --show-current
   # MUST print deployment/aws-hypevault, never main
   git add infra/Dockerfile.backend.aws infra/Dockerfile.frontend.aws infra/docker-compose.aws.yml infra/DEPLOY_AWS.md
   git diff --cached --stat
   git commit -m "deploy: add isolated AWS Docker Compose stack"
   git push -u origin deployment/aws-hypevault
   ```

   If GitHub returns 403, stop. Request collaborator access or push the same
   branch to a fork and open a pull request. Never push directly to `main`.

## EC2 setup (AWS EC2 Instance Connect terminal)

```bash
cd ~/HypeVault
git branch --show-current
git fetch origin deployment/aws-hypevault
git pull --ff-only origin deployment/aws-hypevault

# Generate secrets locally on EC2; never put this file in GitHub or a screenshot.
umask 077
printf 'POSTGRES_PASSWORD=%s\nJWT_SECRET=%s\nPUBLIC_BASE_URL=http://3.106.136.202\n' "$(openssl rand -hex 24)" "$(openssl rand -hex 32)" > .env.aws

sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml config --quiet
sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml up -d --build
sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml ps
sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml exec -T backend alembic upgrade head
curl -i http://localhost/health
```

Open `http://3.106.136.202/` from your browser. EC2's security group must allow
TCP 80 from the internet (and SSH 22 only from known sources). Do not publicly
open PostgreSQL 5432, Redis 6379, backend 8000, or Triton 8001/8002.

For debugging:

```bash
sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml logs --tail=80 backend frontend
sudo docker compose --env-file .env.aws -f infra/docker-compose.aws.yml ps
```

`/health/ready` can return **503** on this CPU instance because Triton is
deliberately offline. `/health` checks only the API process; verify app pages
and database-backed operations separately. Google OAuth and eBay credentials
are not included, so related features may be unavailable.

## Later: GPU inference

Do not confuse the instance name with hardware: **t3.large has no GPU**.
To run Phase 76 / NVIDIA Triton, stop the stack, migrate or resize to a GPU
instance (subject to AWS GPU quota, region capacity and cost), install NVIDIA
drivers and NVIDIA Container Toolkit, copy model files privately, validate
SHA-256 digests, and add a Triton service with a valid model repository. The
provided repository's dev Triton container uses a missing `models/` directory;
starting that container alone does not make Phase 76 available. GPU EC2 is
significantly more expensive than t3.large, so monitor spend and stop it
after the presentation.

Keep large `.pt`, `.npz`, `.joblib` weights and all account secrets **out of Git**.
Do not make your presentation depend on untested GPU inference.
