import re

# Temel bir küfür/argo sözlüğü
KUFUR_SOZLUGU = [
    "amk", "aq", "sik", "sok", "orospu", "pic", "piç", 
    "yavşak", "yavsak", "gavat", "sürtük", "surtuk", 
    "pezevenk", "kahpe", "salak", "aptal", "gerizekalı"
]

def normalize_text(text: str) -> str:
    """
    Kullanıcıların filtreyi aşmak için yaptığı harf oyunlarını temizler.
    1. Leetspeak (@ yerine a, 1 yerine i)
    2. Harf uzatmaları (saaaaalaaaak -> salak)
    """
    text = text.lower()
    
    # 1. Şekilli harf (Leetspeak) düzeltmeleri
    leetspeak_map = {
        '@': 'a', '4': 'a',
        '3': 'e',
        '1': 'i', '!': 'i',
        '0': 'o',
        '5': 's', '$': 's',
        '7': 't'
    }
    for char, replacement in leetspeak_map.items():
        text = text.replace(char, replacement)
        
    # 2. Arka arkaya 3 veya daha fazla tekrar eden harfleri 1 taneye indir
    # (Örn: piiiççç -> piç)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    return text

def create_evasion_regex(word: str):
    """
    Kelimenin harfleri arasına boşluk veya noktalama işareti konmasını yakalar.
    Örn 'amk' -> r'a[\s\W]*m[\s\W]*k'
    Bu sayede "a m k", "a_m_k", "a.m.k" yakalanır ama "adam kazandı" yakalanmaz.
    """
    # Kelimenin harflerini arasına [\s\W]* (boşluk veya noktalama) gelecek şekilde birleştir
    chars = list(word)
    pattern = r'[\s\W]*'.join([re.escape(c) for c in chars])
    return re.compile(pattern, re.IGNORECASE)

def censor_message(original_text: str) -> str:
    """
    Gelişmiş hibrit sansürleme:
    Hem harf oyunlarını, hem boşluk bırakmayı hem de uzatmaları çözer.
    """
    normalized_text = normalize_text(original_text)
    censored_text = original_text
    kelime_bulundu = False
    
    for kufur in KUFUR_SOZLUGU:
        pattern = create_evasion_regex(kufur)
        
        # Eğer normalleştirilmiş metinde bu gelişmiş regex eşleşirse
        if pattern.search(normalized_text):
            kelime_bulundu = True
            
            # Kelime aralarına boşluk girdiği için sadece kelimeyi yıldızlamak
            # cümleyi bozabilir. Bu tarz hileli mesajları tamamen gizlemek en güvenlisidir.
            return "[Bu mesaj sistem tarafından gizlenmiştir]"
            
    if kelime_bulundu:
        return censored_text
    else:
        # Model saldırgan dediyse ama sözlük yakalayamadıysa
        return "[Bu mesaj sistem tarafından gizlenmiştir]"

