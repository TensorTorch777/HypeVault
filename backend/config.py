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
    # Origin check for listing photos: phase76_fusion (DINOv3 + Perception Encoder head, in-process on
    # CUDA; passed benchmark_v6), dinov3_frozen (in-process on CUDA) or dinov2_legacy (the settings
    # below). See inference/live_screening.py.
    live_screening_model: str = "phase76_fusion"
    # Legacy DINOv2 route: triton = NVIDIA Triton gRPC. torch = load LOCAL_MODEL_PATH in-process.
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
    # Listing screening: a photo must show a wristwatch (zero-shot CLIP) and be large enough.
    content_gate_model: str = "openai/clip-vit-base-patch16"
    content_gate_device: str = "cpu"
    content_gate_min_watch_score: float = 0.5
    # Read the dial with OCR after screening and tell the seller when the brand name is unreadable.
    # Informational only: it never changes whether a listing goes live.
    dial_text_check: bool = True
    # Reject files whose signed Content Credentials (C2PA) label says an AI made them (inference.provenance).
    provenance_check: bool = True
    # Reject copies (crops, rescales, screenshots) of AI images already seen (inference.known_ai, SSCD).
    known_ai_check: bool = True
    known_ai_min_similarity: float = 0.65
    # Uploads rejected as AI are added here; a relative path is resolved from backend/.
    known_ai_store: str = "local_known_ai"
    # Phase 76 photos with P(AI) from this value up to its cut are held until the seller sends a proof photo
    # and an admin approves (1.8% of 224 real eBay photos, 0 of 60 AI watch images fell in the band).
    proof_hold_min_p_fake: float = 0.20
    # Also hold (never reject) when the Phase 78 head, trained on screenshot-style copies, calls the photo AI.
    phase78_hold: bool = True
    listing_min_short_side_px: int = 400
    # Deprecated: use inference_min_authentic_confidence. Kept for older .env files (unused in authenticate).
    inference_live_conf_threshold: float = 0.90
    upload_min_side_px: int = 400
    upload_max_bytes: int = 10 * 1024 * 1024
    # Watch identification (search by photo). The gallery is built by scripts/build_watch_gallery.py;
    # a relative path is resolved from the repository root.
    identify_gallery_path: str = "backend/identify/data/watch_gallery.npz"
    identify_min_side_px: int = 200

    # Comma-separated account emails allowed to call /research/*. Empty → nobody. Never set from a request.
    research_user_emails: str = ""

    # Single sign-on: Google accounts from these Workspace domains only (comma-separated).
    sso_allowed_domains: str = "somaiya.edu"
    # Comma-separated emails that become admins on SSO sign-in. Admins also get research access.
    admin_emails: str = "n.vadgaonkar@somaiya.edu"
    # Email/password sign-in, for local development and tests only. Off by default: when on, /auth/register
    # accepts any email domain and the seller role, bypassing the SSO domain rule. Set PASSWORD_LOGIN_ENABLED=true locally.
    password_login_enabled: bool = False

    # Market data (Phase 2a). eBay is read through the official Browse API only; HTML is never scraped.
    ebay_client_id: str = ""
    ebay_client_secret: str = ""
    ebay_environment: str = "production"  # production | sandbox
    ebay_marketplace_id: str = "EBAY_US"
    # Salt for one-way seller pseudonyms. Seller usernames are never stored.
    market_pseudonym_salt: str = "change-me"
    market_cache_ttl_sec: int = 30 * 60


def _email_set(value: str | None) -> frozenset[str]:
    return frozenset(part.strip().casefold() for part in (value or "").split(",") if part.strip())


def research_user_allowlist(raw: str | None = None) -> frozenset[str]:
    if raw is not None:
        return _email_set(raw)
    return _email_set(settings.research_user_emails) | admin_email_set()


def admin_email_set() -> frozenset[str]:
    return _email_set(settings.admin_emails)


def sso_domains() -> frozenset[str]:
    return _email_set(settings.sso_allowed_domains)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
