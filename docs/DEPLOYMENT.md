# Üretim Dağıtım Kontrol Listesi

## Uygulama

- [ ] `APP_ENV=production`
- [ ] `SECRET_KEY` ve `DATA_ENCRYPTION_KEY` secret manager'dan geliyor
- [ ] `ALLOW_MOCK_MODEL=false`, `REQUIRE_MODEL_MANIFEST=true`
- [ ] Model manifesti, tokenizer ve model hash'i doğrulanıyor
- [ ] `alembic upgrade head` başarıyla tamamlandı
- [ ] HTTPS, doğru CORS origin'leri ve ters proxy ayarlandı
- [ ] PostgreSQL, merkezi rate limit ve merkezi log altyapısı kullanılıyor
- [ ] Yönetici hesabı/MFA ve en az yetki politikası yapılandırıldı

## Model

- [ ] Model yeni `turkish-toxic-v2` akışıyla yeniden eğitildi
- [ ] Test setine model veya eşik seçimi sırasında bakılmadı
- [ ] Genel ve dilim bazlı hata oranları onaylandı
- [ ] Canary veya shadow dağıtım yapıldı
- [ ] Drift, latency, inceleme ve itiraz metrikleri için alarm var

## Gizlilik

- [ ] Saklama süresi ve silme prosedürü kurum politikasıyla uyumlu
- [ ] Yedekler şifreli ve süreli
- [ ] Yönetici içerik erişimi denetleniyor
- [ ] Kullanıcıya karar nedeni ve itiraz yolu gösteriliyor

Hazırlık endpoint'i gerçek model zorunlu olduğunda başarısızsa trafik verilmemelidir.
