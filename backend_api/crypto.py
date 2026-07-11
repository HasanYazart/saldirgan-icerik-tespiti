"""Hassas mesaj içeriğini uygulama seviyesinde şifreler."""

from __future__ import annotations

import base64
import hashlib
import hmac

from cryptography.fernet import Fernet, InvalidToken

from .settings import settings


def _fernet_key() -> bytes:
    if settings.data_encryption_key:
        return settings.data_encryption_key.encode("ascii")
    # Yalnızca development için deterministik anahtar. Üretimde Settings bunu engeller.
    return base64.urlsafe_b64encode(hashlib.sha256(settings.secret_key.encode("utf-8")).digest())


fernet = Fernet(_fernet_key())


def encrypt_text(text: str) -> str:
    return fernet.encrypt(text.encode("utf-8")).decode("ascii")


def decrypt_text(token: str | None) -> str | None:
    if not token:
        return None
    try:
        return fernet.decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError):
        return None


def content_fingerprint(text: str) -> str:
    return hmac.new(
        settings.secret_key.encode("utf-8"), text.encode("utf-8"), hashlib.sha256
    ).hexdigest()
