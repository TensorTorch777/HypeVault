# Demo via ngrok

Use this when you want buyers to open the **Next.js** app from the internet while your API and DB still run on your laptop.

## What you need

- [ngrok](https://ngrok.com/) installed and logged in: `ngrok config add-authtoken <token>`
- **Postgres + Redis** running (local or remote)
- **Backend** on `:8000`, **frontend** on `:3000`

You will expose **two** HTTPS URLs (API + web). The free plan usually allows multiple tunnels from one config; if not, run two terminals with two `ngrok http` commands or upgrade.

## 1. Two tunnels (recommended)

Create `ngrok.yml` in your home folder or project (path optional if you use `--config`):

```yaml
version: "2"
authtoken: YOUR_TOKEN   # or omit if already in default config
tunnels:
  hypevault-api:
    addr: 8000
    proto: http
  hypevault-web:
    addr: 3000
    proto: http
```

Start both:

```bash
ngrok start --all --config ./ngrok.yml
```

Note the two **HTTPS** URLs, for example:

- API: `https://abc123.ngrok-free.app`
- Web: `https://xyz789.ngrok-free.app`

## 2. Backend `.env` (repo root)

Set these **before** starting `uvicorn` (restart after edits):

```env
# Public URL of the API tunnel (used for listing image URLs when using local disk uploads)
PUBLIC_API_BASE_URL=https://abc123.ngrok-free.app

# Allow browser on the frontend tunnel to call the API
CORS_ORIGINS=https://xyz789.ngrok-free.app

# Cross-site cookies: frontend and API are different hosts
COOKIE_SECURE=true
COOKIE_SAMESITE=none
```

Keep your normal `DATABASE_URL`, `INFERENCE_*`, etc.

## 3. Frontend `frontend/.env.local`

```env
NEXT_PUBLIC_API_URL=https://abc123.ngrok-free.app
```

Restart `npm run dev` so Next picks this up.

## 4. Google Sign-In (if you use it)

In [Google Cloud Console](https://console.cloud.google.com/) → Credentials → your OAuth client:

- Add **Authorized JavaScript origins**: `https://xyz789.ngrok-free.app`
- Add **Authorized redirect URIs** as required by your flow

ngrok URLs change each time unless you use a **reserved domain** (paid).

## 5. Run the stack

```bash
# Terminal 1 — API (from repo)
cd backend && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
cd frontend && npm run dev

# Terminal 3 — ngrok
ngrok start --all --config ./ngrok.yml
```

Share **only the web tunnel** URL (`https://xyz789.ngrok-free.app`) with demo viewers.

## 6. ngrok browser interstitial

Free tier may show an ngrok landing page on first visit. Viewers click **Visit Site**. For a smoother demo, use a paid reserved domain or ngrok settings to skip where allowed.

## 7. One tunnel only (advanced)

If you can only run **one** ngrok, you’d need a **reverse proxy** (e.g. Caddy) or Next **rewrites** so the browser talks to a single origin. The stock app expects `NEXT_PUBLIC_API_URL` to point at FastAPI; two tunnels is the straightforward path.

## Checklist

| Item | Why |
|------|-----|
| `PUBLIC_API_BASE_URL` = API ngrok URL | Listing images use `/static/local/...` under this host |
| `CORS_ORIGINS` = frontend ngrok URL | Browser allowed to call API |
| `COOKIE_SECURE` + `COOKIE_SAMESITE=none` | Auth cookies on cross-origin XHR |
| `NEXT_PUBLIC_API_URL` = API ngrok URL | Axios hits the tunnel, not `localhost` |

If images 404, double-check `PUBLIC_API_BASE_URL` matches the **same** tunnel you use for the API.
