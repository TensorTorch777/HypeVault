"""FastAPI dependencies for authentication."""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.jwt_handler import decode_token
from config import settings
from database import User, get_db

_log = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    try:
        token: str | None = request.cookies.get(settings.access_cookie_name)
        if not token and credentials is not None and credentials.scheme.lower() == "bearer":
            token = credentials.credentials
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        payload = decode_token(token)
        if payload.get("typ") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id = UUID(sub)
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except HTTPException:
        raise
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        _log.exception("get_current_user_failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_seller(user: Annotated[User, Depends(get_current_user)]) -> User:
    try:
        from database import UserRole

        if user.role != UserRole.seller:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Seller role required")
        return user
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("require_seller_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
