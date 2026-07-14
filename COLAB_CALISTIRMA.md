# Google Colab'da temizleme ve BERT eğitimi

Önce Colab menüsünden **Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU**
seçin. Proje klasörünün tamamını Google Drive'a yükleyin; aşağıdaki hücreleri
sırayla çalıştırın.

## 1. Drive ve proje klasörü

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")

# Yalnızca bu satırı Drive'daki klasörünüze göre değiştirin.
PROJECT_ROOT = Path("/content/drive/MyDrive/DERİN ÖĞRENME YÖNTEMLERİ İLE SOSYAL MEDYADA SALDIRGAN İÇERİK TESPİTİ")
REPO_DIR = PROJECT_ROOT / "github_repo"
os.chdir(REPO_DIR)
print("Çalışma klasörü:", Path.cwd())

# Yeni dosyalar github_repo klasörünün bir üstünde bulunmalı.
required = [
    PROJECT_ROOT / "train_1.csv",
    PROJECT_ROOT / "test_1.csv",
    PROJECT_ROOT / "train_2.csv",
    PROJECT_ROOT / "test_2.csv",
    PROJECT_ROOT / "veri setleri ve url",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Drive'a eksik yüklenen veri/dizin: " + ", ".join(missing))
print("Yeni ve önceki veri dosyaları bulundu.")
```

## 2. BERT ve veri hazırlama bağımlılıkları

```python
!pip install -e ".[colab]"
```

## 3. Tüm CSV'leri temizle, split üret ve kontrol et

```python
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py \
    --prepare-data \
    --prepare-plots \
    --models bert \
    --dry-run
!python tools/audit_dataset.py
```

`--prepare-data`, eğitimden önce proje kökündeki `train_1.csv`, `test_1.csv`,
`train_2.csv`, `test_2.csv` dosyalarını ve `veri setleri ve url/` altındaki dört
eski CSV'yi otomatik bulur. Böylece klasörde önceden duran eski splitler eğitimde
yanlışlıkla kullanılmaz.

`clean=0`; `offensive`, `hate`, `threat`, `targeted_abuse` ve
`sexual_profanity=1` olarak kullanılır. `base_key` ile bağlı sentetik varyasyonlar
aynı `group_id` içinde tutularak farklı split'lere sızmaları engellenir.

## 4. BERT eğitimi

```python
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py \
    --models bert \
    --bert-epochs 3 \
    --bert-batch-size 16
```

T4 belleği yetmezse yalnızca `--bert-batch-size 8` yapın. Colab bağlantısı
kesilirse aynı proje klasöründe şu komut son kaydedilen checkpoint'ten devam eder:

```python
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py \
    --models bert \
    --bert-epochs 3 \
    --bert-batch-size 16 \
    --resume-from-checkpoint
```

Bu komutta yeniden `--prepare-data` yazmanız gerekmez; üçüncü hücre güncel
splitleri üretmiştir. Ham CSV'leri sonradan değiştirirseniz üçüncü hücreyi tekrar
çalıştırın.

Model `modeller/bert_best_model`, ağırlıklar `modeller/bert_best.pt`, eşik ve
ölçümler ise `modeller/model_manifest.json` ile `sonuclar/` altında oluşur.
API klasörüne de kopyalamak isterseniz eğitim komutuna `--deploy-backend` ekleyin.

LSTM/BiLSTM/CNN karşılaştırması ayrıca istenirse TensorFlow bağımlılıklarını
kurup ayrı bir çalıştırmada başlatın:

```python
!pip install -e ".[training,ml]"
!python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py --models lstm,bilstm,cnn
```
