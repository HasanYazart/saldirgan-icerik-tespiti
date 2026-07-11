"""API istek ve yanıt şemaları."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .settings import settings


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=10, max_length=128)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, password: str) -> str:
        if not any(char.isalpha() for char in password) or not any(char.isdigit() for char in password):
            raise ValueError("Parola en az bir harf ve bir rakam içermelidir.")
        if len(password.encode("utf-8")) > 72:
            raise ValueError("Parola UTF-8 biçiminde en fazla 72 bayt olabilir.")
        return password


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class MessageCreate(BaseModel):
    text: str = Field(min_length=1, max_length=settings.max_message_length)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Mesaj boş olamaz.")
        return text


class MessageResponse(BaseModel):
    message_id: int
    status: Literal["clean", "review", "toxic"]
    displayed_text: str
    warning_count: int
    action_taken: Literal["passed", "queued_for_review", "warned", "banned"]
    detection_method: str
    toxicity_score: float = Field(ge=0.0, le=1.0)
    category: str
    severity: str
    model_version: str


class UserStatusResponse(BaseModel):
    user_id: int
    username: str
    role: str
    warning_count: int
    is_banned: bool


class AppealCreate(BaseModel):
    message_id: int = Field(gt=0)
    reason: str = Field(min_length=10, max_length=1000)


class AppealResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    message_id: int
    status: str
    reason: str
    admin_note: str | None
    created_at: datetime


class ReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    admin_note: str = Field(min_length=3, max_length=1000)


class ModerationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    toxicity_score: float
    category: str
    severity: str
    detection_method: str
    action_taken: str
    review_status: str
    model_version: str
    created_at: datetime


class AdminModerationRecord(ModerationRecord):
    content: str | None = None


class AppealDecision(BaseModel):
    decision: Literal["accepted", "rejected"]
    admin_note: str = Field(min_length=3, max_length=1000)
