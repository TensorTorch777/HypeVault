"""Register, login, refresh, and logout."""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response, status
from jose import JWTError
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.deps import get_current_user
from config import settings
from auth.jwt_handler import ACCESS_TOKEN_EXPIRE, REFRESH_TOKEN_EXPIRE, create_access_token, create_refresh_token, decode_token
from auth.models import GoogleAuthRequest, LogoutRequest, RefreshRequest, TokenResponse, UserLogin, UserPublic, UserRegister
from database import User, UserRole, get_db
from redis_client import get_redis

_log = logging.getLogger(__name__)

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__default_rounds=12)


def _hash_password(raw: str) -> str:
    try:
        return pwd_context.hash(raw)
    except Exception as exc:
        _log.exception("hash_password_failed: %s", exc)
        raise


def _verify_password(raw: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(raw, hashed)
    except Exception as exc:
        _log.exception("verify_password_failed: %s", exc)
        return False


def _refresh_redis_key(jti: str) -> str:
    return f"refresh:{jti}"


async def _issue_tokens(user: User) -> TokenResponse:
    access = create_access_token(user_id=str(user.id), role=user.role.value)
    refresh, jti = create_refresh_token(user_id=str(user.id), role=user.role.value)
    redis = await get_redis()
    await redis.set(_refresh_redis_key(jti), str(user.id), ex=int(REFRESH_TOKEN_EXPIRE.total_seconds()))
    return TokenResponse(access_token=access, refresh_token=refresh)


def _set_auth_cookies(response: Response, tokens: TokenResponse) -> None:
    response.set_cookie(
        key=settings.access_cookie_name,
        value=tokens.access_token,
        httponly=True,
        secure=bool(settings.cookie_secure),
        samesite=settings.cookie_samesite,
        max_age=int(ACCESS_TOKEN_EXPIRE.total_seconds()),
        path="/",
    )
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=tokens.refresh_token,
        httponly=True,
        secure=bool(settings.cookie_secure),
        samesite=settings.cookie_samesite,
        max_age=int(REFRESH_TOKEN_EXPIRE.total_seconds()),
        path="/",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(key=settings.access_cookie_name, path="/")
    response.delete_cookie(key=settings.refresh_cookie_name, path="/")


async def _verify_google_id_token(id_token: str) -> str:
    if not settings.google_client_id.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured on server.",
        )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": id_token},
            )
    except Exception as exc:
        _log.exception("google_tokeninfo_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Google verification unavailable")

    if resp.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token")
    data = resp.json()
    aud = str(data.get("aud") or "")
    email = str(data.get("email") or "").strip().lower()
    verified = str(data.get("email_verified") or "").lower() in {"true", "1"}
    if aud != settings.google_client_id.strip() or not email or not verified:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google identity")
    return email


def _infra_unavailable_detail(exc: BaseException) -> str | None:
    """Detect Postgres/Redis connection refused so the UI is not misread as bad password."""
    seen: set[int] = set()
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and len(chain) < 24 and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__

    for e in chain:
        if isinstance(e, ConnectionRefusedError):
            return (
                "Cannot connect to PostgreSQL or Redis (connection refused). "
                "From the repo root run: docker compose -f infra/docker-compose.yml up -d postgres redis "
                "then: cd backend && alembic upgrade head && cd .. && python scripts/seed_database.py"
            )
        if isinstance(e, OSError) and getattr(e, "errno", None) == 111:
            return (
                "Cannot connect to PostgreSQL or Redis (connection refused). "
                "From the repo root run: docker compose -f infra/docker-compose.yml up -d postgres redis "
                "then: cd backend && alembic upgrade head && cd .. && python scripts/seed_database.py"
            )

    low = str(exc).lower()
    if "connection refused" in low or "connect call failed" in low:
        return (
            "Cannot connect to PostgreSQL or Redis. "
            "Start Docker services from repo root: docker compose -f infra/docker-compose.yml up -d postgres redis"
        )
    return None


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register(body: UserRegister, db: Annotated[AsyncSession, Depends(get_db)]) -> User:
    try:
        existing = await db.execute(select(User).where(User.email == str(body.email)))
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
        user = User(
            email=str(body.email),
            password_hash=_hash_password(body.password),
            role=body.role,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("register_failed: %s", exc)
        hint = _infra_unavailable_detail(exc)
        if hint:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=hint)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Registration failed")


@router.post("/login", response_model=TokenResponse)
async def login(
    body: UserLogin,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    try:
        result = await db.execute(select(User).where(User.email == str(body.email)))
        user = result.scalar_one_or_none()
        if user is None or not _verify_password(body.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        tokens = await _issue_tokens(user)
        _set_auth_cookies(response, tokens)
        return tokens
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("login_failed: %s", exc)
        hint = _infra_unavailable_detail(exc)
        if hint:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=hint)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Login failed")


@router.post("/google", response_model=TokenResponse)
async def google_login(
    body: GoogleAuthRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    try:
        email = await _verify_google_id_token(body.id_token)
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(
                email=email,
                password_hash=_hash_password(secrets.token_urlsafe(40)),
                role=UserRole.buyer,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        elif user.role != UserRole.buyer:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Google sign-in is available for buyers only. Use employee email/password login.",
            )
        tokens = await _issue_tokens(user)
        _set_auth_cookies(response, tokens)
        return tokens
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("google_login_failed: %s", exc)
        hint = _infra_unavailable_detail(exc)
        if hint:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=hint)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google login failed")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    body: RefreshRequest = Body(default_factory=RefreshRequest),
) -> TokenResponse:
    try:
        incoming_refresh = body.refresh_token or request.cookies.get(settings.refresh_cookie_name)
        if not incoming_refresh:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token missing",
                headers={"WWW-Authenticate": "Bearer"},
            )
        payload = decode_token(incoming_refresh)
        if payload.get("typ") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        jti = payload.get("jti")
        sub = payload.get("sub")
        if not jti or not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        redis = await get_redis()
        stored = await redis.get(_refresh_redis_key(jti))
        if stored is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token revoked or expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        from uuid import UUID

        user_id = UUID(sub)
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        await redis.delete(_refresh_redis_key(jti))
        access = create_access_token(user_id=str(user.id), role=user.role.value)
        new_refresh, new_jti = create_refresh_token(user_id=str(user.id), role=user.role.value)
        await redis.set(_refresh_redis_key(new_jti), str(user.id), ex=int(REFRESH_TOKEN_EXPIRE.total_seconds()))
        tokens = TokenResponse(access_token=access, refresh_token=new_refresh)
        _set_auth_cookies(response, tokens)
        return tokens
    except HTTPException:
        raise
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        _log.exception("refresh_failed: %s", exc)
        hint = _infra_unavailable_detail(exc)
        if hint:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=hint)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Refresh failed")


@router.post("/logout")
async def logout(request: Request, body: LogoutRequest = Body(default_factory=LogoutRequest)) -> Response:
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    try:
        refresh_token = body.refresh_token or (request.cookies.get(settings.refresh_cookie_name) if request else None)
        if not refresh_token:
            _clear_auth_cookies(response)
            return response
        payload = decode_token(refresh_token)
        jti = payload.get("jti")
        if jti:
            redis = await get_redis()
            await redis.delete(_refresh_redis_key(jti))
    except JWTError:
        _clear_auth_cookies(response)
        return response
    except Exception as exc:
        _log.exception("logout_failed: %s", exc)
    _clear_auth_cookies(response)
    return response


@router.get("/me", response_model=UserPublic)
async def me(user: Annotated[User, Depends(get_current_user)]) -> User:
    try:
        return user
    except Exception as exc:
        _log.exception("me_failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to load profile")
