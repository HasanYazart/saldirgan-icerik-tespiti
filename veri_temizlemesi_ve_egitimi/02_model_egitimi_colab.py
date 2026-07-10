# -*- coding: utf-8 -*-
"""
===============================================================================
 DERİN ÖĞRENME YÖNTEMLERİ İLE SOSYAL MEDYADA SALDIRGAN İÇERİK TESPİTİ
 MODEL EĞİTİMİ - GOOGLE COLAB VERSİYONU
===============================================================================

Bu notebook Google Colab üzerinde GPU ile çalıştırılmak üzere tasarlanmıştır.

Kullanım:
  1. Google Colab'a gidin: https://colab.research.google.com
  2. File > Upload Notebook > Bu dosyayı yükleyin
  3. Runtime > Change runtime type > T4 GPU seçin
  4. Tüm hücreleri çalıştırın (Runtime > Run all)

Eğitilen Modeller:
  - LSTM (Long Short-Term Memory)
  - BiLSTM (Bidirectional LSTM)
  - CNN (1D Convolutional Neural Network)
  - BERT (dbmdz/bert-base-turkish-cased)
===============================================================================
"""

# ============================================================================
# HÜCRE 1: KURULUM VE GITHUB'DAN VERİ ÇEKME
# ============================================================================
# Google Colab'da çalıştırırken bu hücreyi ilk çalıştırın.
# GitHub repo URL'nizi aşağıda güncelleyin!

import os
import subprocess

# ============================
# ⚠️ BURAYI KENDİ REPO URL'NİZLE DEĞİŞTİRİN ⚠️
GITHUB_REPO_URL = "https://github.com/KULLANICI_ADINIZ/saldirgan-icerik-tespiti.git"
# ============================

# Colab'da çalışıp çalışmadığını kontrol et
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    print("✅ Google Colab ortamı tespit edildi!")
except ImportError:
    print("⚠️ Yerel ortamda çalışıyorsunuz.")

# Gerekli kütüphaneleri yükle
if IN_COLAB:
    print("\n📦 Gerekli kütüphaneler yükleniyor...")
    subprocess.run(["pip", "install", "-q", "transformers", "wordcloud"], check=True)
    print("✅ Kütüphaneler yüklendi!")

# GitHub'dan repo'yu klonla
PROJE_KLASORU = "/content/saldirgan-icerik-tespiti"

if IN_COLAB:
    if os.path.exists(PROJE_KLASORU):
        print(f"\n📁 Repo zaten mevcut: {PROJE_KLASORU}")
        # Güncelle
        subprocess.run(["git", "-C", PROJE_KLASORU, "pull"], check=True)
    else:
        print(f"\n📥 GitHub'dan repo klonlanıyor...")
        subprocess.run(["git", "clone", GITHUB_REPO_URL, PROJE_KLASORU], check=True)
        print("✅ Repo klonlandı!")
    
    VERI_KLASORU = os.path.join(PROJE_KLASORU, "veri_setleri")
    SONUC_KLASORU = os.path.join(PROJE_KLASORU, "sonuclar")
    MODEL_KLASORU = os.path.join(PROJE_KLASORU, "modeller")
else:
    # Yerel çalışma
    PROJE_KLASORU = os.path.dirname(os.path.abspath(__file__))
    VERI_KLASORU = os.path.join(PROJE_KLASORU, "veri_setleri")
    SONUC_KLASORU = os.path.join(PROJE_KLASORU, "sonuclar")
    MODEL_KLASORU = os.path.join(PROJE_KLASORU, "modeller")

os.makedirs(SONUC_KLASORU, exist_ok=True)
os.makedirs(MODEL_KLASORU, exist_ok=True)

# Veri dosyalarını kontrol et
for f in ['train.csv', 'valid.csv', 'test.csv']:
    yol = os.path.join(VERI_KLASORU, f)
    if os.path.exists(yol):
        print(f"   ✅ {f} bulundu")
    else:
        print(f"   ❌ {f} BULUNAMADI! Repo URL'nizi kontrol edin.")

print(f"\n📁 Proje klasörü: {PROJE_KLASORU}")
print(f"📁 Veri klasörü:  {VERI_KLASORU}")


# ============================================================================
# HÜCRE 2: KÜTÜPHANELERİ İÇE AKTAR VE GPU KONTROL
# ============================================================================

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100

# TensorFlow
import tensorflow as tf
print(f"\n🔧 TensorFlow versiyon: {tf.__version__}")

# GPU kontrolü
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎮 GPU bulundu: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ GPU bulunamadı! Colab'da Runtime > Change runtime type > T4 GPU seçin.")

# PyTorch (BERT için)
import torch
print(f"🔧 PyTorch versiyon: {torch.__version__}")
print(f"🎮 CUDA kullanılabilir: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Kullanılan cihaz: {device}")


# ============================================================================
# HÜCRE 3: HİPERPARAMETRELER
# ============================================================================

# Keras modelleri (LSTM, BiLSTM, CNN)
MAX_WORDS      = 30000       # Vocabulary boyutu
MAX_SEQ_LEN    = 200         # Maksimum sekans uzunluğu
EMBEDDING_DIM  = 128         # Embedding boyutu
LSTM_UNITS     = 64          # LSTM birim sayısı
CNN_FILTERS    = 128         # CNN filtre sayısı
DROPOUT_RATE   = 0.3         # Dropout oranı
BATCH_SIZE     = 64          # Batch boyutu
EPOCHS         = 10          # Epoch sayısı

# BERT ayarları
BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
BERT_MAX_LEN    = 128        # BERT max token
BERT_BATCH_SIZE = 32         # Colab T4 GPU için uygun
BERT_EPOCHS     = 3          # BERT epoch sayısı
BERT_LR         = 2e-5       # BERT learning rate

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("✅ Hiperparametreler ayarlandı!")


# ============================================================================
# HÜCRE 4: VERİ YÜKLEME
# ============================================================================

print("=" * 60)
print("  VERİ YÜKLEME")
print("=" * 60)

df_train = pd.read_csv(os.path.join(VERI_KLASORU, "train.csv"), encoding='utf-8')
df_valid = pd.read_csv(os.path.join(VERI_KLASORU, "valid.csv"), encoding='utf-8')
df_test  = pd.read_csv(os.path.join(VERI_KLASORU, "test.csv"), encoding='utf-8')

# NaN temizle
for df in [df_train, df_valid, df_test]:
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)

print(f"\n📊 Veri seti boyutları:")
print(f"   Train:      {len(df_train):,} satır")
print(f"   Validation: {len(df_valid):,} satır")
print(f"   Test:       {len(df_test):,} satır")
print(f"   Toplam:     {len(df_train)+len(df_valid)+len(df_test):,} satır")

print(f"\n📊 Sınıf dağılımı (Train):")
print(f"   Normal (0):    {(df_train['label']==0).sum():,} ({(df_train['label']==0).mean()*100:.1f}%)")
print(f"   Saldırgan (1): {(df_train['label']==1).sum():,} ({(df_train['label']==1).mean()*100:.1f}%)")

# Örnek veriler
print(f"\n📋 Örnek veriler:")
for i, row in df_train.head(5).iterrows():
    etiket = "🟢 Normal" if row['label'] == 0 else "🔴 Saldırgan"
    print(f"   {etiket}: {row['text'][:100]}...")


# ============================================================================
# HÜCRE 5: YARDIMCI FONKSİYONLAR
# ============================================================================

def performans_raporu_ciz(y_true, y_pred, y_prob, model_adi):
    """Model performansını görselleştir ve raporla."""
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, accuracy_score, f1_score
    )
    
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"   📊 {model_adi} - TEST SONUÇLARI")
    print(f"{'='*60}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"\n   Classification Report:")
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Saldırgan'])
    print(report)
    
    # Confusion Matrix + ROC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Saldırgan'],
                yticklabels=['Normal', 'Saldırgan'])
    axes[0].set_title(f'{model_adi} - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tahmin', fontsize=12)
    axes[0].set_ylabel('Gerçek', fontsize=12)
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        axes[1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate', fontsize=12)
        axes[1].set_ylabel('True Positive Rate', fontsize=12)
        axes[1].set_title(f'{model_adi} - ROC Eğrisi', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SONUC_KLASORU, f'{model_adi}_sonuclar.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'model': model_adi,
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': roc_auc if y_prob is not None else None
    }


def egitim_grafigi_ciz(history, model_adi):
    """Eğitim/doğrulama loss ve accuracy grafiği."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train', color='#3498db', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', color='#e74c3c', lw=2)
    axes[0].set_title(f'{model_adi} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train', color='#3498db', lw=2)
    axes[1].plot(history.history['val_loss'], label='Validation', color='#e74c3c', lw=2)
    axes[1].set_title(f'{model_adi} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SONUC_KLASORU, f'{model_adi}_egitim.png'), dpi=150, bbox_inches='tight')
    plt.show()


print("✅ Yardımcı fonksiyonlar tanımlandı!")


# ============================================================================
# HÜCRE 6: KERAS TOKENİZATİON (LSTM, BiLSTM, CNN İÇİN ORTAK)
# ============================================================================

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("🔤 Tokenization başlıyor...")

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df_train['text'].values)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(df_train['text'].values),
                            maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
X_valid_seq = pad_sequences(tokenizer.texts_to_sequences(df_valid['text'].values),
                            maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(df_test['text'].values),
                            maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

y_train = df_train['label'].values
y_valid = df_valid['label'].values
y_test  = df_test['label'].values

vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

print(f"   Vocabulary boyutu: {vocab_size:,}")
print(f"   Sekans uzunluğu:   {MAX_SEQ_LEN}")
print(f"   X_train shape:     {X_train_seq.shape}")
print(f"   X_valid shape:     {X_valid_seq.shape}")
print(f"   X_test shape:      {X_test_seq.shape}")
print("✅ Tokenization tamamlandı!")


# ============================================================================
# HÜCRE 7: MODEL 1 - LSTM
# ============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, SpatialDropout1D,
    Bidirectional, Conv1D, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("\n" + "=" * 60)
print("   🚀 MODEL 1: LSTM EĞİTİMİ")
print("=" * 60)

lstm_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQ_LEN),
    SpatialDropout1D(0.2),
    LSTM(LSTM_UNITS, return_sequences=False),
    Dropout(DROPOUT_RATE),
    Dense(32, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

basla = datetime.now()
lstm_history = lstm_model.fit(
    X_train_seq, y_train,
    validation_data=(X_valid_seq, y_valid),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, verbose=1
)
print(f"\n⏱️ LSTM eğitim süresi: {datetime.now() - basla}")

# Eğitim grafiği
egitim_grafigi_ciz(lstm_history, 'LSTM')

# Test
lstm_prob = lstm_model.predict(X_test_seq, verbose=0).flatten()
lstm_pred = (lstm_prob >= 0.5).astype(int)
lstm_sonuc = performans_raporu_ciz(y_test, lstm_pred, lstm_prob, 'LSTM')

# Kaydet
lstm_model.save(os.path.join(MODEL_KLASORU, 'lstm_model.keras'))
print("✅ LSTM modeli kaydedildi!")


# ============================================================================
# HÜCRE 8: MODEL 2 - BiLSTM
# ============================================================================

print("\n" + "=" * 60)
print("   🚀 MODEL 2: BiLSTM EĞİTİMİ")
print("=" * 60)

bilstm_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQ_LEN),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=False)),
    Dropout(DROPOUT_RATE),
    Dense(32, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])

bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
bilstm_model.summary()

basla = datetime.now()
bilstm_history = bilstm_model.fit(
    X_train_seq, y_train,
    validation_data=(X_valid_seq, y_valid),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, verbose=1
)
print(f"\n⏱️ BiLSTM eğitim süresi: {datetime.now() - basla}")

egitim_grafigi_ciz(bilstm_history, 'BiLSTM')

bilstm_prob = bilstm_model.predict(X_test_seq, verbose=0).flatten()
bilstm_pred = (bilstm_prob >= 0.5).astype(int)
bilstm_sonuc = performans_raporu_ciz(y_test, bilstm_pred, bilstm_prob, 'BiLSTM')

bilstm_model.save(os.path.join(MODEL_KLASORU, 'bilstm_model.keras'))
print("✅ BiLSTM modeli kaydedildi!")


# ============================================================================
# HÜCRE 9: MODEL 3 - CNN
# ============================================================================

print("\n" + "=" * 60)
print("   🚀 MODEL 3: CNN EĞİTİMİ")
print("=" * 60)

cnn_model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQ_LEN),
    SpatialDropout1D(0.2),
    Conv1D(CNN_FILTERS, kernel_size=3, activation='relu'),
    Conv1D(CNN_FILTERS, kernel_size=4, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(32, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

basla = datetime.now()
cnn_history = cnn_model.fit(
    X_train_seq, y_train,
    validation_data=(X_valid_seq, y_valid),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, verbose=1
)
print(f"\n⏱️ CNN eğitim süresi: {datetime.now() - basla}")

egitim_grafigi_ciz(cnn_history, 'CNN')

cnn_prob = cnn_model.predict(X_test_seq, verbose=0).flatten()
cnn_pred = (cnn_prob >= 0.5).astype(int)
cnn_sonuc = performans_raporu_ciz(y_test, cnn_pred, cnn_prob, 'CNN')

cnn_model.save(os.path.join(MODEL_KLASORU, 'cnn_model.keras'))
print("✅ CNN modeli kaydedildi!")


# ============================================================================
# HÜCRE 10: MODEL 4 - BERT (TÜRKÇE)
# ============================================================================

print("\n" + "=" * 60)
print("   🚀 MODEL 4: BERT EĞİTİMİ (Türkçe BERT)")
print(f"   Model: {BERT_MODEL_NAME}")
print("=" * 60)

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

# BERT Tokenizer
print("\n🔤 BERT Tokenizer yükleniyor...")
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
print("✅ Tokenizer yüklendi!")

# Dataset sınıfı
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# DataLoader'lar
print("📦 DataLoader'lar oluşturuluyor...")
train_dataset = ToxicDataset(df_train['text'].values, df_train['label'].values, bert_tokenizer, BERT_MAX_LEN)
valid_dataset = ToxicDataset(df_valid['text'].values, df_valid['label'].values, bert_tokenizer, BERT_MAX_LEN)
test_dataset  = ToxicDataset(df_test['text'].values,  df_test['label'].values,  bert_tokenizer, BERT_MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BERT_BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BERT_BATCH_SIZE, shuffle=False)
print(f"   Train batches: {len(train_loader)}")
print(f"   Valid batches: {len(valid_loader)}")
print(f"   Test batches:  {len(test_loader)}")

# BERT Model
print("\n🧠 BERT modeli yükleniyor...")
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)
bert_model.to(device)
print("✅ BERT modeli yüklendi ve GPU'ya aktarıldı!")

# Optimizer & Scheduler
optimizer = torch.optim.AdamW(bert_model.parameters(), lr=BERT_LR, weight_decay=0.01)
total_steps = len(train_loader) * BERT_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Eğitim döngüsü
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_loss = float('inf')

basla = datetime.now()

for epoch in range(BERT_EPOCHS):
    print(f"\n{'─'*60}")
    print(f"   Epoch {epoch+1}/{BERT_EPOCHS}")
    print(f"{'─'*60}")
    
    # === Training ===
    bert_model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"     Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {correct/total:.4f}")
    
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # === Validation ===
    bert_model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_loss = total_loss / len(valid_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"\n   📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"   📉 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
    
    # En iyi modeli kaydet
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(bert_model.state_dict(), os.path.join(MODEL_KLASORU, 'bert_best.pt'))
        print(f"   ✅ En iyi model kaydedildi! (val_loss: {val_loss:.4f})")

bert_sure = datetime.now() - basla
print(f"\n⏱️ BERT eğitim süresi: {bert_sure}")

# BERT Eğitim Grafiği
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, BERT_EPOCHS + 1)

axes[0].plot(epochs_range, train_accs, 'o-', label='Train', color='#3498db', lw=2)
axes[0].plot(epochs_range, val_accs, 'o-', label='Validation', color='#e74c3c', lw=2)
axes[0].set_title('BERT - Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, train_losses, 'o-', label='Train', color='#3498db', lw=2)
axes[1].plot(epochs_range, val_losses, 'o-', label='Validation', color='#e74c3c', lw=2)
axes[1].set_title('BERT - Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=11); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SONUC_KLASORU, 'BERT_egitim.png'), dpi=150, bbox_inches='tight')
plt.show()

# BERT Test
print("\n🧪 BERT test değerlendirmesi...")
bert_model.load_state_dict(torch.load(os.path.join(MODEL_KLASORU, 'bert_best.pt')))
bert_model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        preds = torch.argmax(outputs.logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

bert_sonuc = performans_raporu_ciz(
    np.array(all_labels), np.array(all_preds), np.array(all_probs), 'BERT'
)
print("✅ BERT değerlendirmesi tamamlandı!")


# ============================================================================
# HÜCRE 11: MODEL KARŞILAŞTIRMA
# ============================================================================

print("\n" + "=" * 60)
print("   📊 MODEL KARŞILAŞTIRMA")
print("=" * 60)

tum_sonuclar = [lstm_sonuc, bilstm_sonuc, cnn_sonuc, bert_sonuc]
df_sonuc = pd.DataFrame(tum_sonuclar)

print("\n" + df_sonuc.to_string(index=False))

# Karşılaştırma grafiği
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
modeller = df_sonuc['model'].tolist()
x = np.arange(len(modeller))
colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

# Accuracy
bars1 = axes[0].bar(x, df_sonuc['accuracy'].values, color=colors, edgecolor='black')
axes[0].set_title('Accuracy Karşılaştırma', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_xticks(x); axes[0].set_xticklabels(modeller, fontsize=11)
axes[0].set_ylim(0.5, 1.0)
for bar, val in zip(bars1, df_sonuc['accuracy'].values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)

# F1-Score
bars2 = axes[1].bar(x, df_sonuc['f1_score'].values, color=colors, edgecolor='black')
axes[1].set_title('F1-Score Karşılaştırma', fontsize=14, fontweight='bold')
axes[1].set_ylabel('F1-Score', fontsize=12)
axes[1].set_xticks(x); axes[1].set_xticklabels(modeller, fontsize=11)
axes[1].set_ylim(0.5, 1.0)
for bar, val in zip(bars2, df_sonuc['f1_score'].values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)

# ROC-AUC
roc_vals = df_sonuc['roc_auc'].values
bars3 = axes[2].bar(x, roc_vals, color=colors, edgecolor='black')
axes[2].set_title('ROC-AUC Karşılaştırma', fontsize=14, fontweight='bold')
axes[2].set_ylabel('AUC', fontsize=12)
axes[2].set_xticks(x); axes[2].set_xticklabels(modeller, fontsize=11)
axes[2].set_ylim(0.5, 1.0)
for bar, val in zip(bars3, roc_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(SONUC_KLASORU, 'model_karsilastirma.png'), dpi=150, bbox_inches='tight')
plt.show()

# Sonuçları kaydet
df_sonuc.to_csv(os.path.join(SONUC_KLASORU, 'model_karsilastirma.csv'), index=False)

# En iyi model
en_iyi = df_sonuc.loc[df_sonuc['f1_score'].idxmax()]
print(f"\n🏆 EN İYİ MODEL: {en_iyi['model']}")
print(f"   Accuracy: {en_iyi['accuracy']:.4f}")
print(f"   F1-Score: {en_iyi['f1_score']:.4f}")
print(f"   ROC-AUC:  {en_iyi['roc_auc']:.4f}")


# ============================================================================
# HÜCRE 12: ÖRNEK TAHMİN
# ============================================================================

print("\n" + "=" * 60)
print("   🔮 ÖRNEK TAHMİNLER (En İyi Model: BERT)")
print("=" * 60)

ornek_metinler = [
    "Bu çok güzel bir paylaşım teşekkürler",
    "Seni öldüreceğim lan aptal herif",
    "Bugün hava çok güzel dışarı çıkalım mı",
    "Siktir git buradan gerizekalı",
    "Atatürk çok büyük bir lider",
    "Senin gibi salağı görmedim",
]

bert_model.eval()
print()
for metin in ornek_metinler:
    encoding = bert_tokenizer.encode_plus(
        metin, add_special_tokens=True, max_length=BERT_MAX_LEN,
        padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(outputs.logits, dim=1).item()
        confidence = probs[0][pred].item()
    
    emoji = "🟢" if pred == 0 else "🔴"
    etiket = "Normal" if pred == 0 else "Saldırgan"
    print(f"   {emoji} [{etiket}] (%{confidence*100:.1f}) → \"{metin}\"")


# ============================================================================
# HÜCRE 13: SONUÇLARI GOOGLE DRIVE'A KAYDET (OPSİYONEL)
# ============================================================================

if IN_COLAB:
    kaydet = input("\n💾 Sonuçları Google Drive'a kaydetmek ister misiniz? (e/h): ")
    if kaydet.lower() == 'e':
        from google.colab import drive
        drive.mount('/content/drive')
        
        import shutil
        drive_klasor = '/content/drive/MyDrive/saldirgan_icerik_tespiti_sonuclar'
        os.makedirs(drive_klasor, exist_ok=True)
        
        # Sonuçları kopyala
        shutil.copytree(SONUC_KLASORU, os.path.join(drive_klasor, 'sonuclar'), dirs_exist_ok=True)
        shutil.copytree(MODEL_KLASORU, os.path.join(drive_klasor, 'modeller'), dirs_exist_ok=True)
        
        print(f"\n✅ Sonuçlar Google Drive'a kaydedildi: {drive_klasor}")
    else:
        print("⏭️ Kaydetme atlandı.")

print("\n" + "=" * 60)
print("   ✅ TÜM İŞLEMLER TAMAMLANDI!")
print("=" * 60)
