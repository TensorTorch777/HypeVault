# Deploying the authenticity classifier

## Full stack (frontend + API + DB + local AI) — development

1. **Environment** — Root `.env` should include at least `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, `NEXT_PUBLIC_API_URL`, and the inference block below. **`INFERENCE_MIN_AUTHENTIC_CONFIDENCE`** (default 0.88): model **AUTHENTIC** scores below this are stored and returned as **FAKE** / rejected. **Next.js** reads public vars from `frontend/.env.local` (`NEXT_PUBLIC_API_URL=http://localhost:8000`).

2. **Database** — From repo root:
   ```bash
   docker compose -f infra/docker-compose.yml up -d postgres redis
   bash scripts/run_local_stack.sh
   ```
   The script starts Docker services when possible, runs **Alembic** + **seed**, then prints commands for the API and frontend.

3. **Run processes** (two terminals):
   - API: `cd backend && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
   - Frontend: `cd frontend && npm run dev`
   - Demo logins after seed: `seller@hypevault.demo` / `Seller123`, `buyer@hypevault.demo` / `Buyer123`

4. **`/health/ready`** may stay degraded without Triton or Playwright; **`GET /health`** and listing flows still work with **torch** inference and Postgres + Redis.

---

## Inference backends

| Mode | When to use |
|------|-------------|
| **`INFERENCE_BACKEND=torch`** | Fastest path to production on a **GPU VM**: load `best_model.pt` in-process with PyTorch + timm. |
| **`INFERENCE_BACKEND=triton`** | NVIDIA **Triton** + ONNX (`onnxruntime` backend). Fits Kubernetes / multi-model serving. |

---

## 1. Match shape and weights

Training writes `config.json` next to the checkpoint (e.g. `ml_rtx5080/checkpoints/config.json`). The API **must** use the same values:

- **`INFERENCE_IMG_SIZE`** — same as training `img_size` ( **`504`** for `ml_rtx5080` ViT-B/14, **`518`** for typical Giant runs).
- **`DINOV2_MODEL_NAME`** — same backbone id as training (e.g. **`vit_base_patch14_dinov2.lvd142m`** for your RTX run, or **`dinov2_vitg14_reg`** for Giant).

If these disagree with the checkpoint, scores will be wrong even if the server starts.

---

## 2. Deploy with PyTorch in-process (recommended first)

On a machine with CUDA:

```bash
cd /path/to/HypeVault
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements_inference.txt
```

Copy `best_model.pt` to a stable path (e.g. `models/hypevault_classifier.pt`).

Create or edit **`.env`** in the **repo root**:

```env
INFERENCE_BACKEND=torch
LOCAL_MODEL_PATH=models/hypevault_classifier.pt
DINOV2_MODEL_NAME=vit_base_patch14_dinov2.lvd142m
INFERENCE_IMG_SIZE=504
TORCH_DEVICE=cuda
REPORT_ENFORCE_TRITON=false
```

Run the API from `backend/`:

```bash
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

The stock **`infra/Dockerfile.backend`** only installs `requirements.txt` (no PyTorch). For Docker + GPU you need a **CUDA base image** and `requirements_inference.txt`, or run the API on the host and keep Docker for Postgres/Redis only.

---

## 3. Deploy with Triton (Docker Compose)

1. **Export ONNX** from your checkpoint (sizes and tensor names match Triton config):

   ```bash
   source .venv/bin/activate
   python ml_rtx5080/export_onnx_for_triton.py
   ```

   This writes `ml_rtx5080/checkpoints/dinov2_hypevault.onnx`.

2. **Model repository layout** Triton expects:
   `models/<model_name>/<version>/model.onnx`

   Example for a **504×504** RTX model:

   ```bash
   mkdir -p models/dinov2_classifier/1
   cp ml_rtx5080/checkpoints/dinov2_hypevault.onnx models/dinov2_classifier/1/model.onnx
   ```

3. Edit **`models/dinov2_classifier/config.pbtxt`**: set input dimensions to **`[ 3, 504, 504 ]`** when serving the RTX checkpoint (keep **`[ 3, 518, 518 ]`** only if the ONNX was exported at 518).

4. Start Triton (GPU recommended for throughput; CPU onnxruntime works for smoke tests):

   ```bash
   cd infra && docker compose up -d triton
   ```

   Map ports: **gRPC 18001** on host (see `docker-compose.yml`).

5. Backend **`.env`**:

   ```env
   INFERENCE_BACKEND=triton
   TRITON_HOST=localhost
   TRITON_PORT=18001
   INFERENCE_IMG_SIZE=504
   REPORT_ENFORCE_TRITON=false
   ```

Set **`REPORT_ENFORCE_TRITON=true`** only if you require Triton for compliance-style reporting.

---

## 4. Smoke test

With the API running and a buyer JWT (or your test client), call **`POST /verify/authenticate`** with a JPEG/PNG. Check logs for `local_torch_infer_ok` or `triton_infer_ok` and the returned `verdict` / `confidence`.

---

## 5. Frontend / production

Point **`NEXT_PUBLIC_API_URL`** at your public API URL. Use **HTTPS**, **`COOKIE_SECURE=true`**, and strong **`JWT_SECRET`** in production.
