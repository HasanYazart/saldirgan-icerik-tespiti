"""Eğitim ve canlı tahmin tarafından paylaşılan metin normalizasyonu."""

from __future__ import annotations

import re
import unicodedata


PREPROCESSING_VERSION = "turkish-toxic-v2"

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<![\w@])@[\w.]{2,}(?![\w@])", re.UNICODE)
INVISIBLE_PATTERN = re.compile(
    r"[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e"
    r"\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff\ufff9-\ufffb]"
)
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}", re.UNICODE)

EMOJI_TEXT = {
    "🤡": "palyaço",
    "💩": "dışkı emojisi",
    "🖕": "orta parmak hakaret",
    "🐷": "domuz",
    "🐕": "köpek",
    "🤮": "kusma",
    "💀": "ölüm",
    "🔪": "bıçak",
    "🔫": "silah",
    "👊": "yumruk",
    "🐍": "yılan",
    "🤢": "iğrenme",
}

LEET_MAP = str.maketrans(
    {
        "@": "a",
        "4": "a",
        "3": "e",
        "€": "e",
        "6": "g",
        "9": "g",
        "1": "i",
        "!": "i",
        "|": "i",
        "0": "o",
        "5": "s",
        "$": "s",
        "§": "s",
        "7": "t",
        "+": "t",
    }
)


def translate_known_emojis(text: str) -> str:
    for symbol, replacement in EMOJI_TEXT.items():
        text = text.replace(symbol, f" {replacement} ")
    return text


def normalize_for_model(text: str) -> str:
    """Model girdisini deterministik ve kayıpsız sayılabilecek biçimde normalize et.

    URL ve kullanıcı adları kişisel/oynak değerler yerine sabit belirteçlere
    çevrilir. Noktalama ile kısa mesajlar korunur; böylece eğitim ve üretim
    arasındaki dağılım farkı azaltılır.
    """
    if not isinstance(text, str):
        return ""

    text = INVISIBLE_PATTERN.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = translate_known_emojis(text)
    text = URL_PATTERN.sub(" url ", text)
    text = MENTION_PATTERN.sub(" kullanici ", text)
    text = re.sub(r"(?<!\w)#([\wçğıöşüÇĞİÖŞÜ]+)", r"\1", text)
    text = re.sub(r"(?<!\w)RT(?!\w)", " ", text, flags=re.IGNORECASE)
    text = text.lower().translate(LEET_MAP)
    text = REPEATED_CHAR_PATTERN.sub(r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()
