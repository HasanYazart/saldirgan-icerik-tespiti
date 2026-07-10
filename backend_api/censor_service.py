def censor_message(original_text: str) -> str:
    \"\"\"
    Eğer mesaj saldırgan olarak işaretlenmişse bu fonksiyon çağrılır.
    Kullanıcının tercihi üzerine mesaj tamamen gizlenir.
    \"\"\"
    return "[Bu mesaj sistem tarafından gizlenmiştir]"
