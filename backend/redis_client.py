"""Async Redis client (redis-py asyncio) for sessions, refresh tokens, and cache."""

from __future__ import annotations

import logging

import redis.asyncio as redis

from config import settings

_log = logging.getLogger(__name__)

_redis: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    global _redis
    try:
        if _redis is None:
            _redis = redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
        return _redis
    except Exception as exc:
        _log.exception("redis_connection_failed: %s", exc)
        raise


async def close_redis() -> None:
    global _redis
    try:
        if _redis is not None:
            await _redis.aclose()
    except Exception as exc:
        _log.exception("redis_close_failed: %s", exc)
    finally:
        _redis = None
