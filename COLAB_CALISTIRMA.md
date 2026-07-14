# Google Colab'da Belge (3) yöntemine göre eğitim

Ana deney, belgede karşılaştırılan `LSTM`, `CNN` ve önerilen `CNN → LSTM`
modellerini kullanır. Kelime vektörleri yalnızca train splitinde öğrenilen Word2Vec
ile oluşturulur. BERT, belgenin kuramsal bölümündeki transfer öğrenme karşılaştırması
için ayrıca ve isteğe bağlı çalıştırılır.

## Colab ayarı

Menüden **Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU** seçin.
Proje ana klasörünün tamamını Google Drive'a yükleyin.

## Hücre 1 — Drive'ı bağla

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")

# Drive'da farklı ad kullandıysanız yalnızca bu satırı değiştirin.
PROJECT_ROOT = Path(
    "/content/drive/MyDrive/DERİN ÖĞRENME YÖNTEMLERİ İLE SOSYAL MEDYADA SALDIRGAN İÇERİK TESPİTİ"
)
REPO_DIR = PROJECT_ROOT / "github_repo"
os.chdir(REPO_DIR)

required = [
    PROJECT_ROOT / "train_1.csv",
    PROJECT_ROOT / "test_1.csv",
    PROJECT_ROOT / "train_2.csv",
    PROJECT_ROOT / "test_2.csv",
    PROJECT_ROOT / "veri setleri ve url",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Eksik veri/dizin: " + ", ".join(missing))
print("Çalışma klasörü:", Path.cwd())
print("Yeni ve eski veri kaynakları bulundu.")
```

## Hücre 2 — Bağımlılıkları kur

```python
!pip install -q -e ".[colab]"
```

Kurulumdan sonra Colab yeniden başlatma isterse **Çalışma zamanı > Oturumu yeniden
başlat** deyip Hücre 1'i tekrar çalıştırın.

## Hücre 3 — Tüm verileri temizle ve denetle

```python
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py \
    --prepare-data \
    --prepare-plots \
    --models lstm,cnn,cnn_lstm \
    --dry-run

!python tools/audit_dataset.py
```

Bu adım sekiz CSV'yi birleştirir ve Belge (3), bölüm 3.1'e göre:

- küçük harfe dönüştürür;
- URL/kullanıcı adı, noktalama ve özel karakterleri temizler;
- Türkçe durak kelimeleri kaldırır, fakat anlamı tersine çeviren `değil/yok/hayır`
  sözcüklerini korur;
- boş, geçersiz, çelişkili ve birebir yinelenen satırları ayıklar;
- `base_key` ve yakın-kopya gruplarını bölmeden train/validation/test üretir.

Çıktıda `overlap` ve `group_overlap` değerlerinin tamamı `0` olmalıdır.

## Hücre 4 — Belgedeki ana modelleri eğit

```python
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py \
    --models lstm,cnn,cnn_lstm \
    --word2vec-epochs 10 \
    --keras-epochs 30 \
    --keras-batch-size 64
```

Bu komut Word2Vec'i yalnızca train metinlerinde bir kez eğitir ve aynı vektörleri
üç modelde kullanır. Overfitting'e karşı train-only vocabulary, dondurulmuş Word2Vec,
token dropout, SpatialDropout, L2, label smoothing, sınıf ağırlığı, gradient clipping,
EarlyStopping, ReduceLROnPlateau ve en iyi checkpoint uygulanır.

T4 bellek hatası verirse yalnızca batch boyutunu düşürün:

```python
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py \
    --models lstm,cnn,cnn_lstm \
    --word2vec-epochs 10 \
    --keras-epochs 30 \
    --keras-batch-size 32
```

## Hücre 5 — İsteğe bağlı BERT karşılaştırması

Ana belge deneyinden sonra BERT'i ayrı sonuç klasöründe çalıştırın:

```python
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py \
    --models bert \
    --result-dir sonuclar/bert \
    --model-dir modeller/bert \
    --bert-epochs 3 \
    --bert-batch-size 16
```

Bellek yetmezse `--bert-batch-size 8` kullanın. Bağlantı kesilirse:

```python
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py \
    --models bert \
    --result-dir sonuclar/bert \
    --model-dir modeller/bert \
    --bert-epochs 3 \
    --bert-batch-size 16 \
    --resume-from-checkpoint
```

## Çıktılar

- `sonuclar/model_karsilastirma.csv`: LSTM, CNN ve CNN-LSTM ölçümleri
- `sonuclar/model_metrics.json`: ayrıntılı validation/test metrikleri
- `sonuclar/word2vec_config.json`: Word2Vec deney ayarları
- `modeller/*_best.keras`: en iyi Keras checkpointleri
- `sonuclar/veri_kalitesi/`: temizleme ve veri kalite raporları
- `grafikler/`: veri dağılımı grafikleri

Model seçimini validation F1 ile yapın. Test sonuçlarını yalnızca nihai model
belirlendikten sonra tezde raporlayın.
