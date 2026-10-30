"""Listing photo storage on local disk, served at /static/local/."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from config import settings

_log = logging.getLogger(__name__)

LOCAL_UPLOAD_DIR = Path(__file__).resolve().parent / "local_uploads"
LOCAL_PREFIX = "/static/local/"


def local_public_url(url: str | None) -> str | None:
    """Rebuild a local-disk image URL on the current PUBLIC_API_BASE_URL.

    Rows store an absolute URL from upload time, so a later base (localhost, a new or stopped
    ngrok tunnel) would otherwise leave buyers with broken images. Other URLs pass through unchanged.
    """
    if not url or LOCAL_PREFIX not in url:
        return url
    return f"{settings.public_api_base_url.rstrip('/')}{LOCAL_PREFIX}{url.rsplit(LOCAL_PREFIX, 1)[1]}"


async def upload_file_bytes(
    *,
    key: str,
    body: bytes,
    content_type: str,
) -> str:
    """Save an uploaded photo under LOCAL_UPLOAD_DIR and return its public URL."""
    try:
        LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = key.replace("/", "_")
        path = LOCAL_UPLOAD_DIR / f"{uuid.uuid4()}_{safe_name}"
        path.write_bytes(body)
        return f"{settings.public_api_base_url.rstrip('/')}{LOCAL_PREFIX}{path.name}"
    except Exception as exc:
        _log.exception("upload_failed: %s", exc)
        raise
