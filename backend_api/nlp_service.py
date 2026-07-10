import re
import emoji
from snowballstemmer import TurkishStemmer
from thefuzz import process

stemmer = TurkishStemmer()

KUFUR_SOZLUGU = [
    "amk", "aq", "sik", "sok", "orospu", "pic", "piç", 
    "yavşak", "yavsak", "gavat", "sürtük", "surtuk", 
    "pezevenk", "kahpe", "salak", "aptal", "gerizekalı", "bok"
]

TOXIC_EMOJI_MAP = {
    "🤡": "palyaço",
    "💩": "bok",
    "🖕": "orta parmak",
    "🐷": "domuz"
}

def translate_emojis(text: str) -> str:
    for emj, word in TOXIC_EMOJI_MAP.items():
        text = text.replace(emj, f" {word} ")
    try:
        # Türkçe destekli çeviri (emoji kütüphanesinin güncel sürümlerinde tr dil desteği vardır)
        return emoji.demojize(text, language='tr')
    except:
        return emoji.demojize(text)

def normalize_text(text: str) -> str:
    text = text.lower()
    leetspeak_map = {
        '@': 'a', '4': 'a', '3': 'e', '1': 'i', '!': 'i',
        '0': 'o', '5': 's', '$': 's', '7': 't'
    }
    for char, replacement in leetspeak_map.items():
        text = text.replace(char, replacement)
        
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

def stem_and_correct_typos(text: str) -> str:
    words = text.split()
    processed_words = []
    
    for word in words:
        stemmed_word = stemmer.stemWord(word)
        # Sadece 3 harften uzun kelimelerde typo kontrolü yap (Yanlış eşleşmeleri önlemek için)
        if len(stemmed_word) >= 3:
            match, score = process.extractOne(stemmed_word, KUFUR_SOZLUGU)
            if score >= 85:
                processed_words.append(match)
                continue
        processed_words.append(stemmed_word)
        
    return " ".join(processed_words)

def create_evasion_regex(word: str):
    chars = list(word)
    pattern = r'[\s\W]*'.join([re.escape(c) for c in chars])
    return re.compile(pattern, re.IGNORECASE)

def dictionary_check(text: str) -> bool:
    for kufur in KUFUR_SOZLUGU:
        pattern = create_evasion_regex(kufur)
        if pattern.search(text):
            return True
    return False

def preprocess_text(original_text: str) -> str:
    """
    BERT modeline girmeden önce metni NLP süzgecinden geçirerek temizler.
    """
    text = translate_emojis(original_text)
    text = normalize_text(text)
    text = stem_and_correct_typos(text)
    return text
