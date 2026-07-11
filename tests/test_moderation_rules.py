from backend_api.nlp_service import MessageBuffer, dictionary_check, full_analysis
from backend_api.text_processing import PREPROCESSING_VERSION, normalize_for_model
from backend_api.crypto import decrypt_text
from backend_api.rate_limit import SlidingWindowRateLimiter, build_rate_limiter
from fastapi import HTTPException
import pytest


def test_shared_normalizer_preserves_turkish_and_v_character():
    normalized = normalize_for_model("@Kişi EVVV! https://example.com 🤡")
    assert "kullanici" in normalized
    assert "evv" in normalized
    assert "url" in normalized
    assert "palyaço" in normalized
    assert PREPROCESSING_VERSION == "turkish-toxic-v2"
    assert normalize_for_model(None) == ""
    assert decrypt_text("geçersiz-token") is None


def test_normal_context_is_not_hard_blocked():
    for text in (
        "Bu malzemeyi nereden aldın?",
        "Bugün çok sık görüşüyoruz.",
        "İnternet memesi çok komikti.",
        "Aptalca bir hata yaptım.",
    ):
        assert dictionary_check(text) is False


def test_clear_evasion_is_detected():
    for text in ("s i k", "s.i.k", "@pt@l mısın", "saaaalaaak mısın"):
        assert full_analysis(text, user_id=42)["dictionary_hit"] is True


def test_message_buffer_only_combines_single_letters():
    buffer = MessageBuffer()
    for char in "sala":
        assert buffer.add_and_check(7, char) is False
    assert buffer.add_and_check(7, "k") is True
    assert buffer.add_and_check(8, "normal mesaj") is False


def test_rate_limiter_blocks_excess_requests():
    limiter = build_rate_limiter("", limit=1, window_seconds=60)
    assert isinstance(limiter, SlidingWindowRateLimiter)
    limiter.check("user")
    with pytest.raises(HTTPException) as error:
        limiter.check("user")
    assert error.value.status_code == 429
