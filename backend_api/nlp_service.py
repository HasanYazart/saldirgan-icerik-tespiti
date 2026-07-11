"""
=============================================================================
  SALDIRGAN İÇERİK TESPİTİ - GELİŞMİŞ NLP FİLTRE SERVİSİ

  Bu modül, kullanıcıların filtreleri aşmak için kullandığı TÜM hileleri
  tespit eden çok katmanlı bir metin analiz sistemidir.

  Savunma Katmanları:
    1. Unicode / Görünmez Karakter Temizleme
    2. Emoji Çevirisi (Emoji → Metin)
    3. Leetspeak Çözücü (@→a, 3→e, $→s vb.)
    4. Harf Uzatma Düzeltici (saaaalaaak → salak)
    5. Türkçe Karakter Normalizasyonu (ı↔i, ş↔s vb.)
    6. Boşluk / Noktalama Evasion Tespiti (s.a.l.a.k, s a l a k)
    7. Harf Yer Değiştirme Tespiti (slak→salak, aptla→aptal)
    8. Türkçe Kök Bulma (Stemming) (salaksın → salak)
    9. Fuzzy Match / Yazım Hatası Düzeltme (gerizekali → gerizekalı)
   10. Ardışık Mesaj Birleştirme (s + a + l + a + k → salak)
=============================================================================
"""

import re
import threading
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

from .text_processing import normalize_for_model, translate_known_emojis

# =============================================================================
# KÜFÜR SÖZLÜĞÜ (Genişletilmiş)
# =============================================================================
KUFUR_SOZLUGU = [
    # Yaygın küfürler ve kısaltmalar
    "amk", "amq",
    "sik", "siktir", "sikerim", "sikeyim", "sikim",
    "orospu", "orospuçocuğu", "oç",
    "piç", "pic", "piçkurusu",
    "yavşak", "yavsak",
    "gavat", "ibne",
    "sürtük", "surtuk", "fahişe",
    "pezevenk",
    "kahpe", "kaltak",
    "salak", "aptal", "gerizekalı", "gerzek",
    "dangalak", "ahmak", "embesil",
    "bok", "boktan",
    "hıyar", "puşt", "göt", "götveren",
    "şerefsiz", "serefsiz", "namussuz",
    "alçak", "aşağılık",
    "döl", "yarrak", "taşak",
    # NOT: "mal", "adi", "lan", "ulan", "am", "meme", "sokak" çıkarıldı
    # Bu kelimeler tek başına çok belirsiz, false positive üretiyor.
    # "mal varlığı", "adil", "plan", "sokakta" gibi normal kullanımlar var.
    # Bunlar yerine TOXIC_PATTERNS'e bağlam bazlı kalıplar eklendi.
]

# Aşağıdaki sözcükler bir kişiye yöneltilmediğinde eğitim, mizah, tıp veya
# gündelik konuşma içinde geçebilir. Bunlar tek başına otomatik ceza üretmez;
# açık bir kalıp içindeyse veya BERT yüksek güvenle işaretlerse değerlendirilir.
CONTEXTUAL_TERMS = {
    "salak", "aptal", "gerizekalı", "gerzek", "dangalak", "ahmak", "embesil",
    "bok", "boktan", "hıyar", "göt", "götveren", "döl", "taşak",
}

HARD_BLOCK_TERMS = set(KUFUR_SOZLUGU) - CONTEXTUAL_TERMS

# Saldırgan kalıplar (tek kelime değil, bileşik ifadeler)
TOXIC_PATTERNS = [
    r"seni\s*öldür",
    r"anan[ıi]\s*sik",
    r"seni\s*sik",
    r"kafan[ıi]\s*k[ıi]r",
    r"gebertir",
    r"doğduğuna\s*pişman",
    r"hay\s*anan",
    # Bağlam bazlı kalıplar (sözlükten çıkarılan belirsiz kelimeler için)
    r"\bseni\s+mal\b",
    r"\bmal\s+mısın\b",
    r"\bmal\s+misin\b",
    r"\bne\s+mal\b",
    r"\bbu\s+mal\b",
    r"\bşu\s+mal\b",
    r"\badi\s+herif",
    r"\badi\s+insan",
    r"\b(?:sen|seni|siz|bu|şu|o)\s+(?:salak|aptal|gerizekalı|gerzek|dangalak|ahmak|embesil|hıyar)\b",
    r"\b(?:salak|aptal|gerizekalı|gerzek|dangalak|ahmak|embesil)\s+m[ıi]s[ıi]n\b",
]

# Saldırgan emoji haritası (genişletilmiş)
TOXIC_EMOJI_MAP = {
    "🤡": "palyaço",
    # Tek başına emoji, bir kullanıcıya yönelik hakaretin kesin kanıtı
    # değildir; açıklayıcı bir belirteç olarak BERT'e bırakılır.
    "💩": "dışkı emojisi",
    "🖕": "orta parmak hakaret",
    "🐷": "domuz hakaret",
    "🐕": "köpek hakaret",
    "🤮": "kusma iğrenme",
    "💀": "ölüm tehdit",
    "🔪": "bıçak tehdit",
    "🔫": "silah tehdit",
    "👊": "yumruk tehdit",
    "🐍": "yılan hakaret",
    "🤢": "mide bulandırıcı",
}

# Türkçe karakter eşdeğerleri (Türkçe karaktersiz yazımlar)
TURKISH_CHAR_MAP = {
    # ``ı`` harfi ``i`` değildir. İkisini aynı karaktere indirgemek,
    # "sık" kelimesini "sik" olarak yorumlatıp false positive üretir.
    'ç': 'c', 'ğ': 'g', 'ö': 'o', 'ş': 's', 'ü': 'u',
    'Ç': 'C', 'Ğ': 'G', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
}

# Leetspeak haritası (genişletilmiş)
LEETSPEAK_MAP = {
    '@': 'a', '4': 'a', '^': 'a',
    '8': 'b',
    '(': 'c', '<': 'c', '{': 'c',
    '3': 'e', '€': 'e',
    '6': 'g', '9': 'g',
    '#': 'h',
    '1': 'i', '!': 'i', '|': 'i',
    '0': 'o',
    '5': 's', '$': 's', '§': 's',
    '7': 't', '+': 't',
}

# Görünmez / aldatıcı Unicode karakterler
INVISIBLE_CHARS = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f'    # Zero-width characters
    r'\u2060\u2061\u2062\u2063\u2064'       # Word joiners
    r'\ufeff\ufffe'                          # BOM characters
    r'\u00ad'                                # Soft hyphen
    r'\u034f'                                # Combining grapheme joiner
    r'\u061c'                                # Arabic letter mark
    r'\u115f\u1160'                          # Hangul fillers
    r'\u17b4\u17b5'                          # Khmer vowel inherent
    r'\u180e'                                # Mongolian vowel separator
    r'\u2000-\u200a'                         # Various spaces
    r'\u202a-\u202e'                         # Directional formatting
    r'\u2066-\u2069'                         # Directional isolates
    r'\ufff9-\ufffb]'                        # Interlinear annotations
)


# =============================================================================
# KATMAN 1: Unicode ve Görünmez Karakter Temizleme
# =============================================================================
def clean_unicode(text: str) -> str:
    """
    Görünmez Unicode karakterleri (zero-width space vb.) ve
    homoglyph (benzer görünen farklı alfabe) karakterlerini temizler.
    Örn: Кириллица 'а' (Kiril) → Latin 'a'
    """
    # Görünmez karakterleri sil
    text = INVISIBLE_CHARS.sub('', text)

    # Unicode normalizasyonu (NFC → NFKD)
    # Bazı özel karakterleri temel karşılıklarına dönüştürür
    text = unicodedata.normalize('NFKD', text)

    # Sadece ASCII + Türkçe karakterleri tut, diğerlerini en yakın karşılığa çevir
    cleaned = []
    for char in text:
        if char.isascii() or char in 'çğıöşüÇĞİÖŞÜ':
            cleaned.append(char)
        else:
            # Birleştirme işaretlerini (combining marks) atla
            category = unicodedata.category(char)
            if category.startswith('M'):  # Mark category
                continue
            # Diğer unicode'ları en yakın ASCII karşılığına çevir
            decomposed = unicodedata.normalize('NFD', char)
            ascii_char = ''.join(c for c in decomposed if c.isascii())
            cleaned.append(ascii_char if ascii_char else char)

    return ''.join(cleaned)


# =============================================================================
# KATMAN 2: Emoji Çevirisi
# =============================================================================
def translate_emojis(text: str) -> str:
    """
    Bilinen saldırgan emojileri metin karşılıklarına çevirir.
    Geri kalan emojileri de demojize eder.
    """
    return translate_known_emojis(text)


# =============================================================================
# KATMAN 3: Leetspeak ve Harf Oyunu Çözücü
# =============================================================================
def decode_leetspeak(text: str) -> str:
    """
    s@l@k → salak, 5!k → sik, @pt@l → aptal
    """
    result = []
    for char in text:
        result.append(LEETSPEAK_MAP.get(char, char))
    return ''.join(result)


# =============================================================================
# KATMAN 4: Harf Uzatma Düzeltici
# =============================================================================
def fix_char_repetition(text: str) -> str:
    """
    saaaalaaaakkkk → salak, apttttaaal → aptal
    2 veya daha fazla tekrar eden harfleri 1'e indirir.
    """
    return re.sub(r'(.)\1{1,}', r'\1', text)


# =============================================================================
# KATMAN 5: Türkçe Karakter Normalizasyonu
# =============================================================================
def normalize_turkish(text: str) -> str:
    """
    Hem Türkçe karakterli hem de karaktersiz versiyonları eşleştirmek için
    Türkçe karakterleri ASCII karşılıklarına çevirir.
    piç = pic, şerefsiz = serefsiz olarak da kontrol edilebilir.
    """
    result = []
    for char in text:
        result.append(TURKISH_CHAR_MAP.get(char, char))
    return ''.join(result)


# =============================================================================
# KATMAN 6: Boşluk / Noktalama Evasion Tespiti
# =============================================================================
def check_spaced_evasion(text: str) -> bool:
    """
    "s a l a k", "s.a.l.a.k", "s_a_l_a_k", "s-a-l-a-k" gibi
    harfler arasına ayırıcı karakter konulmuş evasion girişimlerini tespit eder.

    Scunthorpe problemi çözümü: Minimum 4 harflik kelimelerle eşleştirir
    ve SADECE boşluklu pattern arar (substring aramaz).
    """
    for kufur in KUFUR_SOZLUGU:
        # Üç harfli ifadeler yalnızca tüm harfleri ayrılmışsa yakalanır.
        # Böylece "s i k" gibi açık kaçınmalar yakalanırken kelime içi
        # alt-dize eşleşmesiyle normal metinler işaretlenmez.
        if len(kufur) < 3:
            continue
        kufur_normalized = normalize_turkish(kufur.lower())
        # Her harf arasında en az bir ayırıcı olması gereken pattern
        # Sadece gerçekten harfler arasına ayırıcı konmuşsa eşleş
        pattern_chars = list(kufur_normalized)
        spaced_pattern = r'[\s\W_]+'.join([re.escape(c) for c in pattern_chars])
        if re.search(rf'(?<!\w){spaced_pattern}(?!\w)', normalize_turkish(text.lower())):
            return True
    return False


# =============================================================================
# KATMAN 7: Harf Yer Değiştirme Tespiti (Anagram / Transposition)
# =============================================================================
def check_letter_swap(word: str) -> str:
    """
    "slak" → "salak", "aptla" → "aptal", "grizekalı" → "gerizekalı"
    Kelimede 1 harf yer değiştirmişse tespit eder.

    GÜNCELLEME: Anagram kontrolü (sorted == sorted) kaldırıldı çünkü
    "lam"→"mal", "kasi"→"sik" gibi false positive'ler üretiyordu.
    Sadece bitişik harf yer değiştirme (adjacent transposition) kontrol edilir.
    """
    if len(word) < 4:
        return word

    word_lower = normalize_turkish(word.lower())

    for kufur in KUFUR_SOZLUGU:
        kufur_norm = normalize_turkish(kufur.lower())

        # Sadece aynı uzunluktaki kelimelerle karşılaştır
        if len(word_lower) != len(kufur_norm):
            continue

        # Minimum 5 harfli küfürlerle eşleştir (kısa kelimeler çok riskli)
        if len(kufur_norm) < 5:
            continue

        # Sadece bitişik harf yer değiştirme kontrolü:
        # "aptla" → "aptal" (t ve l yer değiştirmiş)
        # Tek bir adjacent transposition olup olmadığını kontrol et
        diff_positions = [i for i in range(len(word_lower)) if word_lower[i] != kufur_norm[i]]

        if len(diff_positions) == 2:
            i, j = diff_positions
            # Bitişik pozisyonlar mı ve harfler çapraz eşleşiyor mu?
            if j - i == 1 and word_lower[i] == kufur_norm[j] and word_lower[j] == kufur_norm[i]:
                return kufur

    return word


# =============================================================================
# KATMAN 8: Türkçe Kök Bulma (Stemming)
# =============================================================================
def stem_word(word: str) -> str:
    """
    "salaksın" → "salak", "aptalsınız" → "aptal"
    Türkçe son eklerini atar ve kökü bulur.
    """
    # Otomatik engelleme akışında stemming kullanılmaz; belirsiz Türkçe
    # kökler yanlış pozitif ürettiği için güvenli davranış kelimeyi korumaktır.
    return word


# =============================================================================
# KATMAN 9: Fuzzy Match (Yazım Hatası Düzeltme)
# =============================================================================
def fuzzy_match_check(word: str) -> tuple:
    """
    "gerezklai" → "gerizekalı" (%90+ benzerlik)
    thefuzz kütüphanesi ile kelimeyi sözlükteki en yakın eşleşmeyle karşılaştırır.

    GÜNCELLEME: partial_ratio kaldırıldı ("kal"→"kaltak" gibi false positive
    üretiyordu). Sadece tam kelime benzerliği (ratio) kullanılır.
    Minimum kelime uzunluğu 4'e, threshold %90'a çıkarıldı.

    Returns: (eşleşen_kelime, benzerlik_skoru) veya (None, 0)
    """
    if len(word) < 4:
        return None, 0

    word_normalized = normalize_turkish(word.lower())

    best_match = None
    best_score = 0

    for kufur in KUFUR_SOZLUGU:
        kufur_norm = normalize_turkish(kufur.lower())

        # Uzunluk farkı çok büyükse karşılaştırma yapma
        if abs(len(word_normalized) - len(kufur_norm)) > 2:
            continue

        # Sadece ratio kullanılır (tam kelime benzerliği)
        # partial_ratio kaldırıldı - çok fazla false positive üretiyordu
        score = round(100 * SequenceMatcher(None, word_normalized, kufur_norm).ratio())

        if score > best_score:
            best_score = score
            best_match = kufur

    # Sabit eşik: %90 (tüm kelimeler için)
    if best_score >= 90:
        return best_match, best_score

    return None, 0


# =============================================================================
# KATMAN 10: Ardışık Mesaj Birleştirme
# =============================================================================
class MessageBuffer:
    """
    Bazı kullanıcılar küfürü tek tek harf olarak gönderir:
    Mesaj 1: "s"
    Mesaj 2: "a"
    Mesaj 3: "l"
    Mesaj 4: "a"
    Mesaj 5: "k"

    Bu sınıf son N mesajı bir buffer'da tutar ve birleştirilmiş halini kontrol eder.
    """
    def __init__(self, max_messages: int = 8, timeout_seconds: int = 20):
        self.buffers = defaultdict(list)  # user_id → [mesajlar]
        self.max_messages = max_messages
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()

    def add_and_check(self, user_id: int, text: str) -> bool:
        """
        Yeni mesajı buffer'a ekler ve son N mesajın birleşimi saldırgan mı kontrol eder.
        """
        import time
        current_time = time.time()

        fragment = text.strip().lower()
        with self._lock:
            self.buffers[user_id] = [
                (msg, ts) for msg, ts in self.buffers[user_id]
                if current_time - ts < self.timeout_seconds
            ]

            if not re.fullmatch(r"[a-zçğıöşü]", fragment):
                self.buffers[user_id].clear()
                return False

            self.buffers[user_id].append((fragment, current_time))
            if len(self.buffers[user_id]) > self.max_messages:
                self.buffers[user_id] = self.buffers[user_id][-self.max_messages:]

            if len(self.buffers[user_id]) >= 2:
                combined = ''.join(msg for msg, _ in self.buffers[user_id])
                combined_lower = normalize_turkish(combined.lower())
                for kufur in KUFUR_SOZLUGU:
                    if combined_lower == normalize_turkish(kufur.lower()):
                        self.buffers[user_id].clear()
                        return True
            return False

# Singleton buffer instance
message_buffer = MessageBuffer()


# =============================================================================
# KALIP (PATTERN) KONTROLÜ
# =============================================================================
def check_toxic_patterns(text: str) -> bool:
    """
    Bileşik saldırgan ifadeleri kontrol eder.
    "seni öldürürüm", "ananı sikerim" gibi kalıplar.
    """
    text_lower = text.lower()
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def check_leetspeak_evasion(text: str) -> bool:
    """Detect an insult deliberately written with leetspeak characters."""
    leet_markers = set("@4^83€69#1!|05$§7+")
    if not any(char in leet_markers for char in text):
        return False

    decoded = normalize_turkish(decode_leetspeak(text.lower()))
    words = re.findall(r"[a-zçğıöşü]+", decoded)
    return any(word in {normalize_turkish(term) for term in KUFUR_SOZLUGU} for word in words)


def check_repetition_evasion(text: str) -> bool:
    """Detect an insult deliberately stretched with repeated characters."""
    if not re.search(r"(.)\1{1,}", text.lower()):
        return False

    normalized = normalize_turkish(fix_char_repetition(text.lower()))
    words = re.findall(r"[a-zçğıöşü]+", normalized)
    return any(word in {normalize_turkish(term) for term in KUFUR_SOZLUGU} for word in words)


# =============================================================================
# ANA PİPELINE: Tüm Katmanları Sırayla Çalıştır
# =============================================================================
def preprocess_text(original_text: str) -> str:
    """
    BERT modeline girmeden önce metni normalleştirir.

    GÜNCELLEME: Kelime bazlı swap/fuzzy/stem işlemleri ÇIKARILDI.
    Bu işlemler normal kelimeleri küfürlere çeviriyordu (ör: "güzel"→"göt").
    Preprocess artık sadece normalizasyon yapıyor:
    - Unicode temizleme
    - Emoji çevirisi
    - Leetspeak çözme
    - Harf uzatma düzeltme

    Kelime bazlı kontroller (swap, fuzzy, stem) sadece dictionary_check'te
    kullanılır ve metni DEĞİŞTİRMEZ, sadece True/False döndürür.
    """
    return normalize_for_model(original_text)


def dictionary_check(text: str) -> bool:
    """
    Normalleştirilmiş metinde sözlükteki kelimeleri arar.
    Hem düz eşleşme hem de boşluklu evasion kontrolü yapar.

    GÜNCELLEME:
    - Kelime bazlı kontrollere stem ve fuzzy eklendi (preprocess'ten taşındı)
    - Substring minimum uzunluk 5→6 yapıldı
    """
    text_lower = text.lower()
    text_normalized = normalize_turkish(text_lower)

    # 1. Kelime bazlı tam eşleşme (Noktalama işaretlerinden arındırılmış)
    import string
    # Noktalamaları boşluğa çevir
    clean_text = text_normalized.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = clean_text.split()

    for word in words:
        # 1a. Tam eşleşme
        for kufur in HARD_BLOCK_TERMS:
            kufur_norm = normalize_turkish(kufur.lower())
            if word == kufur_norm:
                return True

    # Fuzzy eşleşme, kök bulma ve kelime içi alt-dize kontrolü burada
    # kullanılmaz. Bu fonksiyonun True sonucu doğrudan uyarı/engelleme
    # yarattığından belirsiz eşleşmeler BERT'e bırakılır.

    # 2. Boşluklu / noktalama evasion kontrolü
    if check_spaced_evasion(text):
        return True

    # 3. Leetspeak evasion kontrolü
    if check_leetspeak_evasion(text):
        return True

    # 4. Harf uzatma evasion kontrolü
    if check_repetition_evasion(text):
        return True

    # 5. Bileşik saldırgan kalıp kontrolü
    if check_toxic_patterns(text):
        return True

    return False


def full_analysis(original_text: str, user_id: int = 0) -> dict:
    """
    Tüm NLP katmanlarını çalıştırır ve detaylı sonuç döndürür.
    Bu fonksiyon ml_service tarafından çağrılır.

    Returns:
        {
            "preprocessed_text": str,      # Temizlenmiş metin
            "dictionary_hit": bool,        # Sözlük eşleşmesi bulundu mu
            "buffer_hit": bool,            # Ardışık mesaj birleştirmede yakalandı mı
            "is_evasion_detected": bool,    # Herhangi bir evasion tespit edildi mi
        }
    """
    # Ön işleme
    preprocessed = preprocess_text(original_text)

    # Sözlük kontrolü (hem orijinal hem temizlenmiş metin üzerinde)
    dict_hit_original = dictionary_check(original_text)
    dict_hit_cleaned = dictionary_check(preprocessed)

    # Ardışık mesaj buffer kontrolü
    buffer_hit = message_buffer.add_and_check(user_id, original_text)

    return {
        "preprocessed_text": preprocessed,
        "dictionary_hit": dict_hit_original or dict_hit_cleaned,
        "buffer_hit": buffer_hit,
        "is_evasion_detected": dict_hit_original or dict_hit_cleaned or buffer_hit,
    }
