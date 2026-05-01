"""S3 upload helpers and pre-signed URLs (7-day expiry for GET)."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import aioboto3
from botocore.exceptions import ClientError

from config import settings

_log = logging.getLogger(__name__)

LOCAL_UPLOAD_DIR = Path(__file__).resolve().parent / "local_uploads"


def _s3_configured() -> bool:
    return bool(settings.aws_access_key_id and settings.aws_secret_access_key and settings.s3_bucket)


async def upload_file_bytes(
    *,
    key: str,
    body: bytes,
    content_type: str,
) -> str:
    """Upload object to S3; fall back to local disk when AWS is not configured."""
    try:
        if _s3_configured():
            session = aioboto3.Session()
            async with session.client(
                "s3",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            ) as client:
                await client.put_object(
                    Bucket=settings.s3_bucket,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                )
            return f"https://{settings.s3_bucket}.s3.{settings.aws_region}.amazonaws.com/{key}"

        LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = key.replace("/", "_")
        path = LOCAL_UPLOAD_DIR / f"{uuid.uuid4()}_{safe_name}"
        path.write_bytes(body)
        return f"{settings.public_api_base_url.rstrip('/')}/static/local/{path.name}"
    except ClientError as exc:
        _log.exception("s3_upload_client_error: %s", exc)
        raise
    except Exception as exc:
        _log.exception("s3_upload_failed: %s", exc)
        raise


async def generate_presigned_put_url(
    *,
    key: str,
    content_type: str,
    expires_in: int = 3600,
) -> dict[str, Any]:
    """Return a pre-signed PUT URL for browser uploads."""
    try:
        if not _s3_configured():
            return {
                "url": f"{settings.public_api_base_url.rstrip('/')}/uploads/dev-placeholder",
                "key": key,
                "dev": True,
            }
        session = aioboto3.Session()
        async with session.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        ) as client:
            url = await client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": settings.s3_bucket,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=expires_in,
            )
        return {"url": url, "key": key, "dev": False}
    except ClientError as exc:
        _log.exception("s3_presign_put_client_error: %s", exc)
        raise
    except Exception as exc:
        _log.exception("s3_presign_put_failed: %s", exc)
        raise


async def generate_presigned_get_url(*, key: str, expires_in: int = 604800) -> str:
    """Pre-signed GET URL (default 7 days)."""
    try:
        if not _s3_configured():
            return f"{settings.public_api_base_url.rstrip('/')}/static/local/{os.path.basename(key)}"
        session = aioboto3.Session()
        async with session.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        ) as client:
            return await client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": settings.s3_bucket, "Key": key},
                ExpiresIn=expires_in,
            )
    except ClientError as exc:
        _log.exception("s3_presign_get_client_error: %s", exc)
        raise
    except Exception as exc:
        _log.exception("s3_presign_get_failed: %s", exc)
        raise
