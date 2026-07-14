"""Belge (3)'te tarif edilen Türkçe metin ön işleme adımları."""

from __future__ import annotations

import re

from backend_api.text_processing import MENTION_PATTERN, URL_PATTERN, normalize_for_model


PREPROCESSING_VERSION = "belge3-word2vec-v1"

# Belge durak kelimelerin kaldırılmasını ister. Negasyon sözcükleri (değil,
# yok, hayır) saldırganlık anlamını tersine çevirebildiği için görev-özel olarak
# korunur.
TURKISH_STOPWORDS = frozenset(
    """
    acaba ama ancak artık asla aslında az bazı bazen belki ben beni benim beri
    beş bile bir birçok biri birkaç biz bize bizim bu buna bunda bundan bunlar
    bunları bunların çünkü da daha de defa diye eğer en gibi hem hep hepsi her
    herkes hiç için ile ise işte kadar kendi kendine kez ki kim kimi kime kimin
    mı mi mu mü nasıl ne neden nede nerede nereye niçin o olan olarak oldu olmak
    olsa onu onun onlar onları onların öyle pek rağmen sadece sanki sen seni
    senin siz size sizin sonra şu şuna şunda şundan şunlar tarafından üzere var
    ve veya ya yani yine kullanici url rt
    """.split()
)
SPECIAL_CHARACTER_PATTERN = re.compile(
    r"[^0-9a-zçğıöşü\s]", flags=re.IGNORECASE
)
BOUNDARY_PUNCTUATION_PATTERN = re.compile(r"(?<!\w)[^\w\s]+|[^\w\s]+(?!\w)")


def normalize_for_document(value: object) -> str:
    """Küçültme, özel karakter, durak kelime ve boşluk temizliği uygula."""

    if not isinstance(value, str):
        return ""
    # URL ve mention'ı önce işaretle; ardından kelime içi kaçınma yazımlarını
    # (örn. s!ktir) bozmadan cümle sonu/başı noktalamasını kaldır.
    normalized = URL_PATTERN.sub(" url ", value)
    normalized = MENTION_PATTERN.sub(" kullanici ", normalized)
    normalized = BOUNDARY_PUNCTUATION_PATTERN.sub(" ", normalized)
    normalized = normalize_for_model(normalized)
    normalized = SPECIAL_CHARACTER_PATTERN.sub(" ", normalized.lower())
    tokens = [token for token in normalized.split() if token not in TURKISH_STOPWORDS]
    return " ".join(tokens)
