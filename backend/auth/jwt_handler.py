"""JWT access (24h) and refresh (7d) tokens with HS256."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from config import settings

_log = logging.getLogger(__name__)

ACCESS_TOKEN_EXPIRE = timedelta(hours=settings.auth_access_token_expire_hours)
REFRESH_TOKEN_EXPIRE = timedelta(days=settings.auth_refresh_token_expire_days)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(*, user_id: str, role: str) -> str:
    try:
        payload = {
            "sub": user_id,
            "role": role,
            "typ": "access",
            "exp": _now() + ACCESS_TOKEN_EXPIRE,
        }
        return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    except Exception as exc:
        _log.exception("create_access_token_failed: %s", exc)
        raise


def create_refresh_token(*, user_id: str, role: str) -> tuple[str, str]:
    """Return (jwt, jti) for Redis revocation storage."""
    try:
        jti = str(uuid.uuid4())
        payload = {
            "sub": user_id,
            "role": role,
            "typ": "refresh",
            "jti": jti,
            "exp": _now() + REFRESH_TOKEN_EXPIRE,
        }
        token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
        return token, jti
    except Exception as exc:
        _log.exception("create_refresh_token_failed: %s", exc)
        raise


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        _log.debug("jwt_decode_failed: %s", exc)
        raise
    except Exception as exc:
        _log.exception("jwt_decode_unexpected: %s", exc)
        raise
