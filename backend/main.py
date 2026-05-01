"""HypeVault FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from auth.routes import router as auth_router
from config import settings
from database import AsyncSessionLocal
from inference.routes import router as inference_router
from inference.triton_client import triton_ready
from listings.routes import router as listings_router
from redis_client import close_redis, get_redis
from scraper.readiness import check_scraper_readiness
from s3_client import LOCAL_UPLOAD_DIR


def _cors_allow_origins() -> list[str]:
    defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]
    raw = (settings.cors_origins or "").strip()
    if not raw:
        return defaults
    extra = [o.strip() for o in raw.split(",") if o.strip()]
    return list(dict.fromkeys(defaults + extra))

_log = logging.getLogger(__name__)

REQUEST_COUNT = Counter("hypevault_http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("hypevault_request_latency_seconds", "Request latency")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        yield
    finally:
        try:
            await close_redis()
        except Exception as exc:
            _log.exception("redis_close_on_shutdown: %s", exc)


app = FastAPI(title="HypeVault API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    with REQUEST_LATENCY.time():
        response = await call_next(request)
    try:
        REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    except Exception:
        pass
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    try:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": exc.body},
        )
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Validation error"},
        )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    try:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail if isinstance(exc.detail, str) else str(exc.detail)},
            headers=dict(exc.headers) if exc.headers else None,
        )
    except Exception:
        return JSONResponse(status_code=exc.status_code, content={"detail": "Request error"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    _log.exception("unhandled_error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        return {"status": "ok"}
    except Exception as exc:
        _log.exception("health_failed: %s", exc)
        return {"status": "degraded"}


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    """Readiness checks aligned with report deployment path."""
    checks: dict[str, str] = {}
    ok = True
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        _log.warning("health_ready_postgres: %s", exc)
        checks["postgres"] = "error"
        ok = False
    try:
        r = await get_redis()
        if await r.ping():
            checks["redis"] = "ok"
        else:
            checks["redis"] = "error"
            ok = False
    except Exception as exc:
        _log.warning("health_ready_redis: %s", exc)
        checks["redis"] = "error"
        ok = False
    try:
        checks["triton"] = "ok" if await triton_ready() else "error"
        if checks["triton"] != "ok":
            ok = False
    except Exception as exc:
        _log.warning("health_ready_triton: %s", exc)
        checks["triton"] = "error"
        ok = False
    try:
        scraper_ok, scraper_state = check_scraper_readiness()
        checks["scraper"] = "ok" if scraper_ok else f"error:{scraper_state}"
        if not scraper_ok:
            ok = False
    except Exception as exc:
        _log.warning("health_ready_scraper: %s", exc)
        checks["scraper"] = "error"
        ok = False
    payload = {"status": "ready" if ok else "degraded", **checks}
    return JSONResponse(
        status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload,
    )


@app.get("/metrics")
async def metrics() -> Response:
    try:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as exc:
        _log.exception("metrics_failed: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "Metrics unavailable"})


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(inference_router, prefix="/verify", tags=["verify"])
app.include_router(listings_router, prefix="/listings", tags=["listings"])

if LOCAL_UPLOAD_DIR.exists():
    app.mount("/static/local", StaticFiles(directory=str(LOCAL_UPLOAD_DIR)), name="local_uploads")
