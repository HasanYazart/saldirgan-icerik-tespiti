# Güvenlik Politikası

## Desteklenen sürüm

Yalnızca ana dalın güncel sürümü güvenlik düzeltmeleri alır. Güvenlik açığını herkese açık issue olarak paylaşmayın; depo sahibine özel kanaldan iletin.

## Üretim zorunlulukları

- `APP_ENV=production`, `ALLOW_MOCK_MODEL=false`, `REQUIRE_MODEL_MANIFEST=true`
- En az 32 rastgele karakterli `SECRET_KEY`
- `scripts/create_fernet_key.py` ile üretilmiş ayrı `DATA_ENCRYPTION_KEY`
- HTTPS sonlandırma, güvenilir ters proxy ve yalnızca gerekli origin'ler
- Yönetici hesabında benzersiz, güçlü parola; mümkünse dış kimlik sağlayıcı ve MFA
- SQLite yerine yönetilen PostgreSQL; merkezi Redis tabanlı oran sınırı
- Düzenli yedek, anahtar rotasyonu, erişim logu ve bağımlılık taraması

JWT veya şifreleme anahtarı repoya commit edilmemelidir. `DATA_ENCRYPTION_KEY` kaybedilirse kayıtlı mesajlar çözülemez; anahtar açığa çıkarsa yeni anahtara geçilip saklanan hassas içerikler yeniden şifrelenmelidir.

## Veri minimizasyonu

Ham mesaj sütunu varsayılan olarak kapalıdır. Mesajlar şifreli biçimde `MESSAGE_RETENTION_DAYS` kadar tutulur. Analitik için tersine çevrilemeyen HMAC parmak izi kullanılır. Yönetici paneli içeriği yalnızca yetkili kullanıcıya çözer.

## Bilinen sınırlar

Geliştirme modundaki oran sınırlayıcı tek süreç içindir. Üretimde `REDIS_URL` zorunludur ve bütün worker'lar merkezi sayacı kullanır. JWT iptali için merkezi denylist bulunmadığından erişim token süresi kısa tutulmalıdır.
