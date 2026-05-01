"""Pydantic schemas for auth."""

from __future__ import annotations

import re
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator

from database import UserRole


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: UserRole = UserRole.buyer

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Za-z]", v) or not re.search(r"\d", v):
            raise ValueError("Password must include letters and numbers")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: str = Field(min_length=20, max_length=5000)
    role: UserRole = UserRole.buyer


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str | None = None


class UserPublic(BaseModel):
    id: UUID
    email: EmailStr
    role: UserRole

    model_config = {"from_attributes": True}


class LogoutRequest(BaseModel):
    refresh_token: str | None = None
