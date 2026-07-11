# Derin Öğrenme ile Türkçe Saldırgan İçerik Tespiti

[![CI](https://github.com/HasanYazart/saldirgan-icerik-tespiti/actions/workflows/ci.yml/badge.svg)](https://github.com/HasanYazart/saldirgan-icerik-tespiti/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB)
![API](https://img.shields.io/badge/API-FastAPI-009688)

Türkçe sosyal medya mesajlarını kural tabanlı kaçınma kontrolleri ve BERT ile değerlendiren, insan incelemesi ve itiraz akışı içeren FastAPI uygulaması.

## Proje durumu

| Kontrol | Sonuç |
|---|---|
| Otomatik test | 14 test başarılı |
| Backend coverage | %75,09 (minimum %70) |
| Python ve JavaScript derleme kontrolü | Başarılı |
| Alembic migration | Yeni ve eski SQLite şemasında başarılı |
| Veri sızıntısı kontrolü | Train/valid/test birebir metin kesişimi 0 |
| Model çalışma modu | Manifestli v2 model yoksa güvenli `rules_only/degraded` |

Depodaki eski BERT ağırlığı `turkish-toxic-v2` ön işleme sürümüyle eğitilmediğinden üretim modeli olarak yüklenmez. Gerçek BERT modu için veri seti yeni pipeline ile yeniden üretilmeli ve model yeniden eğitilmelidir; doğrulanmamış modelle sessiz tahmin yapılmaz.

## Özellikler

- Türkçe BERT sınıflandırması ve doğrulama verisiyle kalibre edilen karar eşiği
- Leetspeak, görünmez karakter, harf ayırma ve ardışık mesaj kaçınma kontrolleri
- JWT kimlik doğrulama; kullanıcı ve yönetici rol ayrımı
- Belirsiz kararlar için insan inceleme kuyruğu ve kullanıcı itirazı
- Mesaj içeriğini Fernet ile şifreli, süreli saklama; ham içerik varsayılan olarak kapalı
- Model manifesti, SHA-256 doğrulaması ve eğitim–servis ön işleme sürüm kontrolü
- Alembic migration, Docker, otomatik test ve GitHub Actions

## Mimari

```text
İstemci
  │ JWT
  ▼
FastAPI ──► oran sınırı ──► ortak normalizasyon
  │                              │
  │                              ├─► yüksek kesinlikli kurallar
  │                              └─► sürümlenmiş BERT
  │
  ├─► karar: geç / inceleme / uyarı
  ├─► şifreli moderasyon kaydı
  └─► yönetici incelemesi + itiraz
```

## Hızlı başlangıç

### Docker ile geliştirme modu

1. `.env.example` dosyasını `.env` olarak kopyalayın.
2. `SECRET_KEY` değerini uzun ve rastgele bir değerle değiştirin.
3. `DATA_ENCRYPTION_KEY` üretin:

```powershell
python scripts/create_fernet_key.py
```

4. Uygulamayı başlatın:

```powershell
docker compose up --build
```

Arayüz `http://localhost:8000`, geliştirme API belgesi `http://localhost:8000/docs` adresindedir.

Model bağlanmadığında geliştirme ortamı `rules_only` modunda çalışır ve sağlık durumu `degraded` döner. Bu mod üretim için uygun değildir.

### Yerel Python kurulumu

Python 3.11 veya 3.12 gerekir.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ml,dev]"
alembic upgrade head
uvicorn backend_api.main:app --reload
```

`.env` dosyası otomatik okunmaz; yerel çalıştırmada değişkenleri kabuğa aktarın veya Uvicorn'u `uvicorn backend_api.main:app --env-file .env` ile başlatın.

## Model artefactı

Üretim dağıtımında `MODEL_DIR` aşağıdaki yapıyı içermelidir:

```text
model/
├── bert_best.pt
├── bert_best_model/
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── ...
└── model_manifest.json
```

Manifest alanları [model_manifest.example.json](backend_api/model_manifest.example.json) dosyasında gösterilir. Eğitim betiği model, tokenizer, eşik ve manifesti birlikte üretir. Eski `bert_best.pt`, yeni `turkish-toxic-v2` normalizasyonuyla yeniden eğitilmeden üretimde kullanılmamalıdır.

Üretim ayarları:

```dotenv
APP_ENV=production
ALLOW_MOCK_MODEL=false
REQUIRE_MODEL_MANIFEST=true
MODEL_DIR=/app/model
DATABASE_URL=postgresql+psycopg://user:password@postgres/moderation
REDIS_URL=redis://redis:6379/0
```

## Veri hazırlama ve eğitim

Ham veri dosyalarını varsayılan olarak `ham_veri/` içine koyun:

```text
ham_veri/
├── turkish_toxic_language.csv
├── train.csv
├── valid.csv
└── test.csv
```

Başka konum için `RAW_DATA_DIR` ayarlanabilir.

```powershell
pip install -e ".[training]"
python veri_temizlemesi_ve_egitimi/01_veri_temizleme.py
python tools/audit_dataset.py
python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py
```

Temizleme betiği kısa mesajları korur, aynı metne verilmiş çelişkili etiketlerde işlemi durdurur ve kaynak+etiket dağılımını bölümler arasında korur. Eğitim betiği test verisini model/ensemble ağırlığı seçmek için kullanmaz.

## API özeti

| Endpoint | Yetki | Amaç |
|---|---|---|
| `POST /api/auth/register` | Açık | Kullanıcı oluşturur |
| `POST /api/auth/login` | Açık | JWT üretir |
| `GET /api/me/status` | Kullanıcı | Kendi durumunu döndürür |
| `POST /api/chat/send` | Kullanıcı | Mesajı analiz eder |
| `POST /api/appeals` | Kullanıcı | Moderasyon kararına itiraz eder |
| `GET /api/admin/reviews` | Yönetici | İnceleme kuyruğunu döndürür |
| `POST /api/admin/reviews/{id}` | Yönetici | İnceleme kararı verir |
| `GET/POST /api/admin/appeals` | Yönetici | İtirazları yönetir |
| `GET /api/health` | Açık | Canlılık ve model durumunu döndürür |
| `GET /api/ready` | Açık | Dağıtım hazır olma durumunu döndürür |

İlk yöneticiyi oluşturmak için `.env` içinde `ADMIN_USERNAME` ve güçlü bir `ADMIN_PASSWORD` birlikte ayarlanır. Üretimde kayıt kapatılacaksa `ALLOW_REGISTRATION=false` kullanılır.

## Test

```powershell
pip install -e ".[dev]"
python -m compileall -q backend_api tests
pytest --cov=backend_api --cov-report=term-missing
```

CI her push ve pull request'te derleme, test, coverage ve migration smoke testini çalıştırır.

## Güvenlik ve model sorumluluğu

Bu sistem tek başına geri döndürülemez kullanıcı yaptırımı vermek için tasarlanmamıştır. Otomatik ban varsayılan olarak kapalıdır. Dil, lehçe, alıntı, mizah ve bağlam kaynaklı yanlış kararlar insan incelemesi ve itirazla düzeltilmelidir.

- Güvenlik politikası: [SECURITY.md](SECURITY.md)
- Model kartı: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- Veri kartı: [docs/DATA_CARD.md](docs/DATA_CARD.md)
- Dağıtım kontrol listesi: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Lisans notu

Kaynak kod için henüz açık kaynak lisansı seçilmemiştir. Veri setlerinin ve `dbmdz/bert-base-turkish-cased` modelinin kendi lisans/atıf koşulları dağıtımdan önce ayrıca doğrulanmalıdır.
