"""SQLAlchemy 2.0 veri modelleri."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user", index=True)
    warning_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_banned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list[MessageLog]] = relationship(back_populates="user")
    appeals: Mapped[list[Appeal]] = relationship(back_populates="user")


class MessageLog(Base):
    __tablename__ = "message_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    original_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    encrypted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    displayed_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_toxic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    toxicity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="clean", index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="none", index=True)
    detection_method: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    action_taken: Mapped[str] = mapped_column(String(32), nullable=False, default="passed", index=True)
    review_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="not_required", index=True
    )
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False, default="rules-only")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    user: Mapped[User] = relationship(back_populates="messages")
    appeals: Mapped[list[Appeal]] = relationship(back_populates="message")


class Appeal(Base):
    __tablename__ = "appeals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("message_logs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open", index=True)
    admin_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="appeals")
    message: Mapped[MessageLog] = relationship(back_populates="appeals")
