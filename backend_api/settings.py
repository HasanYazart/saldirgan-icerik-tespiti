"""Uygulama yapılandırması.

Tüm ortama bağlı değerler tek noktadan okunur. Üretimde güvenli olmayan
varsayılanlarla uygulamanın açılmasına izin verilmez.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Saldırgan İçerik Tespiti API")
    app_env: str = os.getenv("APP_ENV", "development").lower()
    secret_key: str = os.getenv("SECRET_KEY", "development-only-change-me")
    data_encryption_key: str = os.getenv("DATA_ENCRYPTION_KEY", "")
    admin_username: str = os.getenv("ADMIN_USERNAME", "")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")
    access_token_minutes: int = int(os.getenv("ACCESS_TOKEN_MINUTES", "60"))
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{(PROJECT_DIR / 'toxic_chat.db').as_posix()}"
    )
    redis_url: str = os.getenv("REDIS_URL", "")
    allowed_origins: list[str] = field(
        default_factory=lambda: _as_list("ALLOWED_ORIGINS", "http://localhost:8000")
    )
    allow_registration: bool = _as_bool("ALLOW_REGISTRATION", True)
    allow_mock_model: bool = _as_bool("ALLOW_MOCK_MODEL", True)
    require_model_manifest: bool = _as_bool("REQUIRE_MODEL_MANIFEST", False)
    store_raw_messages: bool = _as_bool("STORE_RAW_MESSAGES", False)
    message_retention_days: int = int(os.getenv("MESSAGE_RETENTION_DAYS", "30"))
    max_message_length: int = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
    moderation_rate_limit: int = int(os.getenv("MODERATION_RATE_LIMIT", "30"))
    rate_limit_window_seconds: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    auto_ban_warning_count: int = int(os.getenv("AUTO_BAN_WARNING_COUNT", "0"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(BASE_DIR))).resolve()

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def validate(self) -> None:
        if self.is_production and self.secret_key == "development-only-change-me":
            raise RuntimeError("Üretimde SECRET_KEY zorunludur.")
        if self.is_production and not self.data_encryption_key:
            raise RuntimeError("Üretimde DATA_ENCRYPTION_KEY zorunludur.")
        if bool(self.admin_username) != bool(self.admin_password):
            raise RuntimeError("ADMIN_USERNAME ve ADMIN_PASSWORD birlikte ayarlanmalıdır.")
        if self.is_production and self.allow_mock_model:
            raise RuntimeError("Üretimde ALLOW_MOCK_MODEL=false olmalıdır.")
        if self.is_production and not self.redis_url:
            raise RuntimeError("Üretimde merkezi oran sınırı için REDIS_URL zorunludur.")
        if not 1 <= self.max_message_length <= 20_000:
            raise RuntimeError("MAX_MESSAGE_LENGTH 1-20000 aralığında olmalıdır.")
        if self.auto_ban_warning_count < 0:
            raise RuntimeError("AUTO_BAN_WARNING_COUNT negatif olamaz.")


settings = Settings()
settings.validate()
