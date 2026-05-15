"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (parent of backend/) — single .env at project root works for uvicorn, alembic, and scripts
_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = _REPO_ROOT  # project root (parent of backend/)
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://hypevault:hypevault@localhost:5432/hypevault"
    redis_url: str = "redis://localhost:6379"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    google_client_id: str = ""
    access_cookie_name: str = "hv_access"
    refresh_cookie_name: str = "hv_refresh"
    cookie_secure: bool = False
    cookie_samesite: str = "lax"
    auth_access_token_expire_hours: int = 24
    auth_refresh_token_expire_days: int = 7

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-south-1"
    s3_bucket: str = "hypevault-listings"
    sqs_queue_url: str = ""

    triton_host: str = "localhost"
    triton_port: int = 8001

    # Local dev: serve uploads when S3 is not configured
    public_api_base_url: str = "http://localhost:8000"

    # Price comparison scraper (Playwright). Use regional host, e.g. https://www.chrono24.in
    chrono24_search_base: str = "https://www.chrono24.in"

    # Comma-separated origins for browser dev (e.g. Next on :3001). Empty → sensible localhost defaults.
    cors_origins: str = ""

    # ── Inference backend ───────────────────────────────────────────────────
    # triton = NVIDIA Triton gRPC (production). torch = load LOCAL_MODEL_PATH in-process.
    inference_backend: str = "triton"
    # Path to .pt from training (`torch.save({"model_state": ...})`). Relative paths are under repo root.
    local_model_path: str = ""
    dinov2_model_name: str = "dinov2_vitg14_reg"
    # cuda | cpu | cuda:0 — empty = auto (cuda if available)
    torch_device: str = ""
    triton_model_name: str = "dinov2_classifier"
    triton_input_name: str = "input__0"
    triton_output_name: str = "output__0"

    # Must match training `img_size` (e.g. 518 for Giant runs, 504 for ml_rtx5080 ViT-B/14).
    inference_img_size: int = 518

    # Report-strict gates
    report_enforce_triton: bool = True
    # Model AUTHENTIC branch reports P(authentic). Below this → stored and returned as FAKE (rejected).
    inference_min_authentic_confidence: float = 0.88
    # Deprecated: use inference_min_authentic_confidence. Kept for older .env files (unused in authenticate).
    inference_live_conf_threshold: float = 0.90
    upload_min_side_px: int = 400
    upload_max_bytes: int = 10 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
