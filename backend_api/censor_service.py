import re

# Temel bir küfür/argo sözlüğü (Buraya eklenebilir)
KUFUR_SOZLUGU = [
    "amk", "aq", "sik", "sok", "orospu", "pic", "piç", 
    "yavşak", "yavsak", "gavat", "sürtük", "surtuk", 
    "pezevenk", "kahpe", "salak", "aptal", "gerizekalı"
]

def censor_message(original_text: str) -> str:
    \"\"\"
    Mesajı sansürler.
    1. İçinde sözlükteki küfürler varsa onları *** yapar.
    2. Eğer model saldırgan dediyse ama sözlükte küfür bulamadıysa, 
       mesajı tamamen gizler.
    \"\"\"
    censored_text = original_text.lower()
    kelime_bulundu = False
    
    for kufur in KUFUR_SOZLUGU:
        # Regex ile tam kelime veya kelime içinde geçenleri yıldızla
        pattern = re.compile(re.escape(kufur), re.IGNORECASE)
        if pattern.search(censored_text):
            kelime_bulundu = True
            censored_text = pattern.sub("*" * len(kufur), censored_text)
            
    if kelime_bulundu:
        return censored_text
    else:
        # Yapay zeka saldırgan buldu ama içinde bizim bildiğimiz açık bir küfür yok 
        # (Örn: "Seni doğduğuna pişman ederim" gibi bir tehdit).
        # Bu durumda kelime gizleyemeyeceğimiz için mesajı tamamen kapatıyoruz.
        return "[Bu mesaj sistem tarafından gizlenmiştir]"

