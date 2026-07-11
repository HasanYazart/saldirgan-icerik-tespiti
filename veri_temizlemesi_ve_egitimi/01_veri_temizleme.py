# -*- coding: utf-8 -*-
"""
===============================================================================
 DERİN ÖĞRENME YÖNTEMLERİ İLE SOSYAL MEDYADA SALDIRGAN İÇERİK TESPİTİ
 01 - VERİ TEMİZLEME & KEŞİFSEL VERİ ANALİZİ (EDA)
===============================================================================

Bu script iki farklı Türkçe saldırgan içerik veri setini:
  1. HuggingFace - turkish_toxic_language.csv  (~77.800 satır)
  2. Kaggle - train.csv / test.csv / valid.csv  (~53.000 satır)

yükler, temizler, birleştirir ve eğitime hazır hale getirir.

Çalıştırma:
  python 01_veri_temizleme.py

Gerekli kütüphaneler:
  pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
===============================================================================
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split

SCRIPT_KLASORU = os.path.dirname(os.path.abspath(__file__))
PROJE_KLASORU = os.path.abspath(os.path.join(SCRIPT_KLASORU, ".."))
if PROJE_KLASORU not in sys.path:
    sys.path.insert(0, PROJE_KLASORU)

from backend_api.text_processing import normalize_for_model

warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

# Matplotlib Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# YAPILANDIRMA
# ============================================================================

# Dosya yolları
VERI_KLASORU = os.getenv("RAW_DATA_DIR", os.path.join(PROJE_KLASORU, "ham_veri"))
CIKTI_KLASORU = os.path.join(PROJE_KLASORU, "veri_setleri")
GRAFIK_KLASORU = os.path.join(PROJE_KLASORU, "grafikler")

# Temizleme parametreleri
MIN_KELIME_SAYISI = 1        # Kısa sosyal medya mesajları kritik örneklerdir
MAX_KARAKTER_SAYISI = 1500   # Bu sayıdan fazla karakter kırpılır
TEST_ORANI = 0.15            # Test seti oranı
VALID_ORANI = 0.10           # Validation seti oranı
RANDOM_SEED = 42

# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def klasor_olustur(klasor_yolu):
    """Klasör yoksa oluşturur."""
    os.makedirs(klasor_yolu, exist_ok=True)
    print(f"  📁 Klasör hazır: {klasor_yolu}")


def metin_temizle(text):
    """
    Türkçe metin temizleme fonksiyonu.
    
    Adımlar:
      1. NaN/None kontrolü
      2. URL'leri kaldır
      3. @mention'ları kaldır
      4. #hashtag'lerden # işaretini kaldır (kelimeyi koru)
      5. RT (retweet) etiketini kaldır
      6. Emoji ve özel karakterleri kaldır (Türkçe harfler korunur)
      7. Fazla boşlukları tek boşluğa düşür
      8. Baş ve son boşlukları temizle
      9. Küçük harfe çevir
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    return normalize_for_model(text)


def kelime_say(text):
    """Metindeki kelime sayısını döndürür."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return 0
    return len(text.split())


# ============================================================================
# 1. ADIM: VERİ SETLERİNİ YÜKLE
# ============================================================================

print("=" * 70)
print("  ADIM 1: VERİ SETLERİNİ YÜKLEME")
print("=" * 70)

# --- HuggingFace Veri Seti ---
print("\n📂 HuggingFace - turkish_toxic_language.csv yükleniyor...")
df_toxic = pd.read_csv(
    os.path.join(VERI_KLASORU, "turkish_toxic_language.csv"),
    encoding='utf-8'
)
print(f"   ✅ Yüklendi: {df_toxic.shape[0]:,} satır, {df_toxic.shape[1]} sütun")
print(f"   Sütunlar: {df_toxic.columns.tolist()}")

# --- Kaggle Veri Seti ---
print("\n📂 Kaggle - train/test/valid.csv yükleniyor...")
df_train = pd.read_csv(os.path.join(VERI_KLASORU, "train.csv"), encoding='utf-8')
df_test = pd.read_csv(os.path.join(VERI_KLASORU, "test.csv"), encoding='utf-8')
df_valid = pd.read_csv(os.path.join(VERI_KLASORU, "valid.csv"), encoding='utf-8')

df_kaggle = pd.concat([df_train, df_test, df_valid], ignore_index=True)
print(f"   ✅ Train:  {df_train.shape[0]:,} satır")
print(f"   ✅ Test:   {df_test.shape[0]:,} satır")
print(f"   ✅ Valid:  {df_valid.shape[0]:,} satır")
print(f"   ✅ Toplam: {df_kaggle.shape[0]:,} satır")
print(f"   Sütunlar: {df_kaggle.columns.tolist()}")


# ============================================================================
# 2. ADIM: ÖN İNCELEME (TEMİZLEME ÖNCESİ)
# ============================================================================

print("\n" + "=" * 70)
print("  ADIM 2: ÖN İNCELEME (TEMİZLEME ÖNCESİ)")
print("=" * 70)

# --- HuggingFace ---
print("\n📊 HuggingFace Veri Seti:")
print(f"   Null değerler:\n{df_toxic.isnull().sum().to_string()}")
print(f"   Duplikat satır: {df_toxic.duplicated().sum()}")
print(f"   is_toxic dağılımı:")
print(f"     0 (Normal):    {(df_toxic['is_toxic']==0).sum():,} ({(df_toxic['is_toxic']==0).mean()*100:.1f}%)")
print(f"     1 (Saldırgan): {(df_toxic['is_toxic']==1).sum():,} ({(df_toxic['is_toxic']==1).mean()*100:.1f}%)")
print(f"   target dağılımı:")
for t, c in df_toxic['target'].value_counts().items():
    print(f"     {t:15s}: {c:6,d} ({c/len(df_toxic)*100:.1f}%)")

# --- Kaggle ---
print("\n📊 Kaggle Veri Seti:")
print(f"   Null değerler:\n{df_kaggle.isnull().sum().to_string()}")
print(f"   Duplikat satır: {df_kaggle.duplicated().sum()}")
print(f"   label dağılımı:")
print(f"     0 (Normal):    {(df_kaggle['label']==0).sum():,} ({(df_kaggle['label']==0).mean()*100:.1f}%)")
print(f"     1 (Saldırgan): {(df_kaggle['label']==1).sum():,} ({(df_kaggle['label']==1).mean()*100:.1f}%)")


# ============================================================================
# 3. ADIM: SÜTUN UYUMLULAŞTIRMA
# ============================================================================

print("\n" + "=" * 70)
print("  ADIM 3: SÜTUN UYUMLULAŞTIRMA")
print("=" * 70)

# HuggingFace: text, target, source, is_toxic -> text, label
df_hf = df_toxic[['text', 'is_toxic']].copy()
df_hf.columns = ['text', 'label']
df_hf['kaynak'] = 'huggingface'
print(f"   ✅ HuggingFace: {len(df_hf):,} satır -> [text, label, kaynak]")

# Kaggle: id, text, label -> text, label
df_kg = df_kaggle[['text', 'label']].copy()
df_kg['kaynak'] = 'kaggle'
print(f"   ✅ Kaggle:      {len(df_kg):,} satır -> [text, label, kaynak]")


# ============================================================================
# 4. ADIM: METİN TEMİZLEME
# ============================================================================

print("\n" + "=" * 70)
print("  ADIM 4: METİN TEMİZLEME")
print("=" * 70)

print("\n🧹 HuggingFace metinleri temizleniyor...")
df_hf['text_temiz'] = df_hf['text'].apply(metin_temizle)
print("   ✅ Tamamlandı")

print("\n🧹 Kaggle metinleri temizleniyor...")
df_kg['text_temiz'] = df_kg['text'].apply(metin_temizle)
print("   ✅ Tamamlandı")

# Temizleme önce/sonra örnekleri
print("\n📋 Temizleme örnekleri (Önce → Sonra):")
for i, (_, row) in enumerate(df_kg[df_kg['text'] != df_kg['text_temiz']].head(5).iterrows()):
    print(f"\n   Örnek {i+1}:")
    print(f"   ÖNCE : {row['text'][:120]}")
    print(f"   SONRA: {row['text_temiz'][:120]}")


# ============================================================================
# 5. ADIM: FİLTRELEME
# ============================================================================

print("\n" + "=" * 70)
print("  ADIM 5: FİLTRELEME")
print("=" * 70)

def filtrele_ve_raporla(df, isim):
    """Boş, çok kısa ve duplikat satırları filtreler."""
    baslangic = len(df)
    
    # Boş metinleri çıkar
    df = df[df['text_temiz'].str.strip() != ''].copy()
    bos_cikarilan = baslangic - len(df)
    
    # Çok kısa metinleri çıkar
    df['kelime_sayisi'] = df['text_temiz'].apply(kelime_say)
    onceki = len(df)
    df = df[df['kelime_sayisi'] >= MIN_KELIME_SAYISI].copy()
    kisa_cikarilan = onceki - len(df)
    
    # Çok uzun metinleri kırp
    uzun_mask = df['text_temiz'].str.len() > MAX_KARAKTER_SAYISI
    uzun_sayisi = uzun_mask.sum()
    df.loc[uzun_mask, 'text_temiz'] = df.loc[uzun_mask, 'text_temiz'].str[:MAX_KARAKTER_SAYISI]
    
    # Duplikat text'leri çıkar
    onceki = len(df)
    df = df.drop_duplicates(subset=['text_temiz']).copy()
    duplikat_cikarilan = onceki - len(df)
    
    print(f"\n   📊 {isim}:")
    print(f"     Başlangıç:           {baslangic:,}")
    print(f"     Boş çıkarılan:       {bos_cikarilan:,}")
    print(f"     Kısa çıkarılan (<{MIN_KELIME_SAYISI}): {kisa_cikarilan:,}")
    print(f"     Uzun kırpılan (>{MAX_KARAKTER_SAYISI}): {uzun_sayisi:,}")
    print(f"     Duplikat çıkarılan:  {duplikat_cikarilan:,}")
    print(f"     Kalan:               {len(df):,}")
    
    return df

df_hf = filtrele_ve_raporla(df_hf, "HuggingFace")
df_kg = filtrele_ve_raporla(df_kg, "Kaggle")


# ============================================================================
# 6. ADIM: VERİ SETLERİNİ BİRLEŞTİR
# ============================================================================

print("\n" + "=" * 70)
print("  ADIM 6: VERİ SETLERİNİ BİRLEŞTİRME")
print("=" * 70)

# Birleştir
df_birlesik = pd.concat([
    df_hf[['text_temiz', 'label', 'kaynak']],
    df_kg[['text_temiz', 'label', 'kaynak']]
], ignore_index=True)

# Sütun adını düzelt
df_birlesik.rename(columns={'text_temiz': 'text'}, inplace=True)

# Birleşik duplikatları çıkar
onceki = len(df_birlesik)
etiket_sayilari = df_birlesik.groupby('text')['label'].nunique()
celiskili_metinler = etiket_sayilari[etiket_sayilari > 1].index
if len(celiskili_metinler):
    klasor_olustur(CIKTI_KLASORU)
    celiski_raporu = df_birlesik[df_birlesik['text'].isin(celiskili_metinler)].sort_values('text')
    celiski_yolu = os.path.join(CIKTI_KLASORU, 'etiket_celiskileri.csv')
    celiski_raporu.to_csv(celiski_yolu, index=False, encoding='utf-8')
    raise ValueError(f"{len(celiskili_metinler)} çelişkili metin bulundu. İnceleyin: {celiski_yolu}")

df_birlesik = df_birlesik.drop_duplicates(subset=['text']).copy()
print(f"   Birleşik duplikat çıkarılan: {onceki - len(df_birlesik):,}")

print(f"\n   ✅ Birleşik veri seti: {len(df_birlesik):,} satır")
print(f"   Sınıf dağılımı:")
print(f"     0 (Normal):    {(df_birlesik['label']==0).sum():,} ({(df_birlesik['label']==0).mean()*100:.1f}%)")
print(f"     1 (Saldırgan): {(df_birlesik['label']==1).sum():,} ({(df_birlesik['label']==1).mean()*100:.1f}%)")
print(f"   Kaynak dağılımı:")
for k, c in df_birlesik['kaynak'].value_counts().items():
    print(f"     {k:15s}: {c:,}")


# ============================================================================
# 7. ADIM: TRAIN / TEST / VALIDATION SPLIT
# ============================================================================

print("\n" + "=" * 70)
print("  ADIM 7: TRAIN / TEST / VALIDATION SPLIT")
print("=" * 70)

# Stratified split (sınıf dengesi korunarak)
X = df_birlesik[['text', 'kaynak']]
y = df_birlesik['label']
strata = y.astype(str) + '_' + df_birlesik['kaynak'].astype(str)

# Önce train+valid ve test ayır
X_train_valid, X_test, y_train_valid, y_test = train_test_split(
    X, y,
    test_size=TEST_ORANI,
    random_state=RANDOM_SEED,
    stratify=strata
)

# Sonra train ve valid ayır
valid_ratio_adjusted = VALID_ORANI / (1 - TEST_ORANI)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_valid, y_train_valid,
    test_size=valid_ratio_adjusted,
    random_state=RANDOM_SEED,
    stratify=(y_train_valid.astype(str) + '_' + X_train_valid['kaynak'].astype(str))
)

print(f"\n   📊 Bölümleme sonuçları:")
print(f"     Train:      {len(X_train):,} satır ({len(X_train)/len(X)*100:.1f}%)")
print(f"     Validation: {len(X_valid):,} satır ({len(X_valid)/len(X)*100:.1f}%)")
print(f"     Test:       {len(X_test):,} satır ({len(X_test)/len(X)*100:.1f}%)")

print(f"\n   Sınıf dağılımı (Train):")
print(f"     0: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
print(f"     1: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")

# DataFrame'leri oluştur
df_train_final = X_train.copy(); df_train_final['label'] = y_train
df_valid_final = X_valid.copy(); df_valid_final['label'] = y_valid
df_test_final  = X_test.copy(); df_test_final['label'] = y_test


# ============================================================================
# 8. ADIM: TEMİZLENMİŞ VERİYİ KAYDET
# ============================================================================

print("\n" + "=" * 70)
print("  ADIM 8: TEMİZLENMİŞ VERİYİ KAYDETME")
print("=" * 70)

klasor_olustur(CIKTI_KLASORU)

df_train_final.to_csv(os.path.join(CIKTI_KLASORU, "train.csv"), index=False, encoding='utf-8')
df_valid_final.to_csv(os.path.join(CIKTI_KLASORU, "valid.csv"), index=False, encoding='utf-8')
df_test_final.to_csv(os.path.join(CIKTI_KLASORU, "test.csv"), index=False, encoding='utf-8')
df_birlesik.to_csv(os.path.join(CIKTI_KLASORU, "birlesik_tum_veri.csv"), index=False, encoding='utf-8')

print(f"   ✅ train.csv  → {len(df_train_final):,} satır")
print(f"   ✅ valid.csv  → {len(df_valid_final):,} satır")
print(f"   ✅ test.csv   → {len(df_test_final):,} satır")
print(f"   ✅ birlesik_tum_veri.csv → {len(df_birlesik):,} satır")


# ============================================================================
# 9. ADIM: KEŞİFSEL VERİ ANALİZİ (EDA)
# ============================================================================

print("\n" + "=" * 70)
print("  ADIM 9: KEŞİFSEL VERİ ANALİZİ (EDA)")
print("=" * 70)

klasor_olustur(GRAFIK_KLASORU)

# --- Grafik 1: Sınıf Dağılımı ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
sinif_sayilari = df_birlesik['label'].value_counts()
labels_pie = ['Normal (0)', 'Saldırgan (1)']
colors_pie = ['#2ecc71', '#e74c3c']
axes[0].pie(sinif_sayilari.values, labels=labels_pie, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Sınıf Dağılımı (Pasta Grafik)', fontsize=14, fontweight='bold')

# Bar chart
sinif_sayilari.plot(kind='bar', ax=axes[1], color=colors_pie, edgecolor='black')
axes[1].set_title('Sınıf Dağılımı (Çubuk Grafik)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Etiket', fontsize=12)
axes[1].set_ylabel('Sayı', fontsize=12)
axes[1].set_xticklabels(['Normal (0)', 'Saldırgan (1)'], rotation=0)
for i, v in enumerate(sinif_sayilari.values):
    axes[1].text(i, v + 200, f'{v:,}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASORU, '01_sinif_dagilimi.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 01_sinif_dagilimi.png kaydedildi")


# --- Grafik 2: Metin Uzunluğu Dağılımı ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_birlesik['metin_uzunlugu'] = df_birlesik['text'].str.len()
df_birlesik['kelime_sayisi'] = df_birlesik['text'].str.split().str.len()

# Karakter uzunluğu
for label, color, isim in [(0, '#2ecc71', 'Normal'), (1, '#e74c3c', 'Saldırgan')]:
    subset = df_birlesik[df_birlesik['label'] == label]['metin_uzunlugu']
    axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=isim, edgecolor='black')
axes[0].set_title('Metin Uzunluğu Dağılımı (Karakter)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Karakter Sayısı', fontsize=12)
axes[0].set_ylabel('Frekans', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].set_xlim(0, 800)

# Kelime sayısı
for label, color, isim in [(0, '#2ecc71', 'Normal'), (1, '#e74c3c', 'Saldırgan')]:
    subset = df_birlesik[df_birlesik['label'] == label]['kelime_sayisi']
    axes[1].hist(subset, bins=50, alpha=0.6, color=color, label=isim, edgecolor='black')
axes[1].set_title('Metin Uzunluğu Dağılımı (Kelime)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Kelime Sayısı', fontsize=12)
axes[1].set_ylabel('Frekans', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].set_xlim(0, 150)

plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASORU, '02_metin_uzunlugu.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 02_metin_uzunlugu.png kaydedildi")


# --- Grafik 3: Kaynak Dağılımı ---
fig, ax = plt.subplots(figsize=(8, 5))
kaynak_sayilari = df_birlesik['kaynak'].value_counts()
kaynak_sayilari.plot(kind='bar', ax=ax, color=['#3498db', '#e67e22'], edgecolor='black')
ax.set_title('Kaynak Bazında Veri Dağılımı', fontsize=14, fontweight='bold')
ax.set_xlabel('Kaynak', fontsize=12)
ax.set_ylabel('Sayı', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(kaynak_sayilari.values):
    ax.text(i, v + 200, f'{v:,}', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASORU, '03_kaynak_dagilimi.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 03_kaynak_dagilimi.png kaydedildi")


# --- Grafik 4: Box Plot - Metin Uzunlukları ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x='label', y='metin_uzunlugu', data=df_birlesik, ax=axes[0],
            palette=['#2ecc71', '#e74c3c'])
axes[0].set_title('Karakter Uzunluğu (Box Plot)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Etiket (0=Normal, 1=Saldırgan)', fontsize=12)
axes[0].set_ylabel('Karakter Sayısı', fontsize=12)
axes[0].set_ylim(0, 800)

sns.boxplot(x='label', y='kelime_sayisi', data=df_birlesik, ax=axes[1],
            palette=['#2ecc71', '#e74c3c'])
axes[1].set_title('Kelime Sayısı (Box Plot)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Etiket (0=Normal, 1=Saldırgan)', fontsize=12)
axes[1].set_ylabel('Kelime Sayısı', fontsize=12)
axes[1].set_ylim(0, 150)

plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASORU, '04_boxplot_uzunluk.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 04_boxplot_uzunluk.png kaydedildi")


# --- Grafik 5: Word Cloud ---
try:
    from wordcloud import WordCloud
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Normal metinler
    normal_text = ' '.join(df_birlesik[df_birlesik['label'] == 0]['text'].tolist())
    wc_normal = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Greens',
        max_words=100,
        collocations=False
    ).generate(normal_text)
    axes[0].imshow(wc_normal, interpolation='bilinear')
    axes[0].set_title('Normal Metinler - Word Cloud', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Saldırgan metinler
    saldirgan_text = ' '.join(df_birlesik[df_birlesik['label'] == 1]['text'].tolist())
    wc_saldirgan = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='Reds',
        max_words=100,
        collocations=False
    ).generate(saldirgan_text)
    axes[1].imshow(wc_saldirgan, interpolation='bilinear')
    axes[1].set_title('Saldırgan Metinler - Word Cloud', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFIK_KLASORU, '05_wordcloud.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 05_wordcloud.png kaydedildi")
    
except ImportError:
    print("   ⚠️ wordcloud kütüphanesi yüklü değil. Word cloud oluşturulamadı.")
    print("      Yüklemek için: pip install wordcloud")


# --- Grafik 6: En Sık Kullanılan Kelimeler ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (label, title, color) in enumerate([(0, 'Normal', '#2ecc71'), (1, 'Saldırgan', '#e74c3c')]):
    all_words = ' '.join(df_birlesik[df_birlesik['label'] == label]['text'].tolist()).split()
    
    # Türkçe stop words (temel)
    stop_words = {'bir', 'bu', 'de', 'da', 've', 'ile', 'için', 'mi', 'mı', 'mu', 'mü',
                  'ne', 'ben', 'sen', 'biz', 'siz', 'o', 'var', 'yok', 'ki', 'ama',
                  'daha', 'en', 'gibi', 'kadar', 'sonra', 'olan', 'olarak', 'ya',
                  'hem', 'her', 'hiç', 'çok', 'bile', 'şey', 'ise', 'olan'}
    
    filtered_words = [w for w in all_words if w not in stop_words and len(w) > 2]
    word_counts = Counter(filtered_words).most_common(20)
    
    words, counts = zip(*word_counts)
    axes[idx].barh(range(len(words)), counts, color=color, edgecolor='black')
    axes[idx].set_yticks(range(len(words)))
    axes[idx].set_yticklabels(words, fontsize=10)
    axes[idx].invert_yaxis()
    axes[idx].set_title(f'En Sık Kelimeler - {title}', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Frekans', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASORU, '06_en_sik_kelimeler.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 06_en_sik_kelimeler.png kaydedildi")


# --- Grafik 7: Train/Test/Valid Dağılımı ---
fig, ax = plt.subplots(figsize=(8, 5))
split_sayilari = [len(df_train_final), len(df_valid_final), len(df_test_final)]
split_labels = [f'Train\n({len(df_train_final):,})', 
                f'Validation\n({len(df_valid_final):,})', 
                f'Test\n({len(df_test_final):,})']
colors_split = ['#3498db', '#f39c12', '#e74c3c']
ax.pie(split_sayilari, labels=split_labels, autopct='%1.1f%%',
       colors=colors_split, startangle=90, textprops={'fontsize': 12})
ax.set_title('Train / Validation / Test Dağılımı', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAFIK_KLASORU, '07_split_dagilimi.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ 07_split_dagilimi.png kaydedildi")


# ============================================================================
# 10. ADIM: ÖZET
# ============================================================================

print("\n" + "=" * 70)
print("  ÖZET")
print("=" * 70)

print(f"""
   📁 Temizlenmiş veriler: {CIKTI_KLASORU}
      ├── train.csv          ({len(df_train_final):,} satır)
      ├── valid.csv          ({len(df_valid_final):,} satır)
      ├── test.csv           ({len(df_test_final):,} satır)
      └── birlesik_tum_veri.csv ({len(df_birlesik):,} satır)

   📊 Grafikler: {GRAFIK_KLASORU}
      ├── 01_sinif_dagilimi.png
      ├── 02_metin_uzunlugu.png
      ├── 03_kaynak_dagilimi.png
      ├── 04_boxplot_uzunluk.png
      ├── 05_wordcloud.png
      ├── 06_en_sik_kelimeler.png
      └── 07_split_dagilimi.png

   ✅ Veri temizleme tamamlandı! 
   ➡️  Sonraki adım: python 02_model_egitimi.py
""")
