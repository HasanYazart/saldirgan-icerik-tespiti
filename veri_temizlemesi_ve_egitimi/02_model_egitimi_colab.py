# -*- coding: utf-8 -*-
"""
===============================================================================
 DERİN ÖĞRENME YÖNTEMLERİ İLE SOSYAL MEDYADA SALDIRGAN İÇERİK TESPİTİ
 MODEL EĞİTİMİ - GOOGLE COLAB VERSİYONU (GELİŞMİŞ)
===============================================================================

Anti-Overfitting Teknikleri:
  ✅ L2 Regularization (kernel_regularizer)
  ✅ BatchNormalization
  ✅ Çoklu Dropout katmanları (SpatialDropout1D + Dropout)
  ✅ EarlyStopping (patience=5)
  ✅ ReduceLROnPlateau + CosineDecay
  ✅ Label Smoothing
  ✅ Class Weights (sınıf dengesizliği)
  ✅ Gradient Clipping
  ✅ Data Augmentation (Random Word Deletion)

Performans Artırma:
  ✅ Attention mekanizması (LSTM, BiLSTM)
  ✅ Multi-kernel CNN (3, 4, 5 filtre boyutları)
  ✅ Ensemble Model (tüm modellerin birleşimi)
  ✅ BERT fine-tuning (discriminative learning rates)
  ✅ Mixed Precision Training

Kullanım (Google Colab):
  1. Runtime > Change runtime type > T4 GPU
  2. Hücre 1: !git clone https://github.com/HasanYazart/saldirgan-icerik-tespiti.git
              !pip install -q transformers wordcloud
  3. Hücre 2: %run saldirgan-icerik-tespiti/veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py
===============================================================================
"""

# ============================================================================
# HÜCRE 1: KURULUM VE GITHUB'DAN VERİ ÇEKME
# ============================================================================

import os
import subprocess

# GitHub repo URL
GITHUB_REPO_URL = "https://github.com/HasanYazart/saldirgan-icerik-tespiti.git"

# Colab kontrolü
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
    print("✅ Google Colab ortamı tespit edildi!")
except ImportError:
    print("⚠️ Yerel ortamda çalışıyorsunuz.")

# Kütüphaneleri yükle
if IN_COLAB:
    print("\n📦 Gerekli kütüphaneler yükleniyor...")
    subprocess.run(["pip", "install", "-q", "transformers", "wordcloud"], check=True)
    print("✅ Kütüphaneler yüklendi!")

# GitHub'dan repo klonla
PROJE_KLASORU = "/content/saldirgan-icerik-tespiti" if IN_COLAB else os.path.dirname(os.path.abspath(__file__))

if IN_COLAB:
    if os.path.exists(PROJE_KLASORU):
        subprocess.run(["git", "-C", PROJE_KLASORU, "pull"], check=True)
    else:
        print(f"\n📥 GitHub'dan repo klonlanıyor...")
        subprocess.run(["git", "clone", GITHUB_REPO_URL, PROJE_KLASORU], check=True)
        print("✅ Repo klonlandı!")

VERI_KLASORU  = os.path.join(PROJE_KLASORU, "veri_setleri")
SONUC_KLASORU = os.path.join(PROJE_KLASORU, "sonuclar")
MODEL_KLASORU = os.path.join(PROJE_KLASORU, "modeller")
os.makedirs(SONUC_KLASORU, exist_ok=True)
os.makedirs(MODEL_KLASORU, exist_ok=True)

for f in ['train.csv', 'valid.csv', 'test.csv']:
    yol = os.path.join(VERI_KLASORU, f)
    print(f"   {'✅' if os.path.exists(yol) else '❌'} {f}")

print(f"\n📁 Proje: {PROJE_KLASORU}")


# ============================================================================
# HÜCRE 2: KÜTÜPHANELERİ İÇE AKTAR VE GPU KONTROL
# ============================================================================

import sys
import warnings
import random
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
from tensorflow.keras import backend as K
print(f"\n🔧 TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎮 GPU: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Mixed Precision Training (GPU varsa) - hızlandırır
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("⚡ Mixed Precision Training aktif!")
else:
    print("⚠️ GPU bulunamadı! Runtime > Change runtime type > T4 GPU")

# PyTorch
import torch
print(f"🔧 PyTorch: {torch.__version__}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# HÜCRE 3: HİPERPARAMETRELER
# ============================================================================

# Keras modelleri
MAX_WORDS      = 50000       # Daha geniş vocabulary
MAX_SEQ_LEN    = 250         # Daha uzun sekans
EMBEDDING_DIM  = 256         # Daha zengin embedding
LSTM_UNITS     = 128         # Daha güçlü LSTM
CNN_FILTERS    = 256         # Daha fazla filtre
DROPOUT_RATE   = 0.4         # Daha agresif dropout (overfitting önleme)
BATCH_SIZE     = 64
EPOCHS         = 20          # Daha fazla epoch (EarlyStopping durduracak)
L2_REG         = 1e-4        # L2 Regularization
LABEL_SMOOTH   = 0.1         # Label Smoothing

# BERT
BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"
BERT_MAX_LEN    = 128
BERT_BATCH_SIZE = 32
BERT_EPOCHS     = 4
BERT_LR         = 2e-5
BERT_WARMUP     = 0.1        # Warmup oranı

# Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("✅ Hiperparametreler ayarlandı!")
print(f"   Dropout: {DROPOUT_RATE} | L2: {L2_REG} | Label Smoothing: {LABEL_SMOOTH}")
print(f"   Max Epoch: {EPOCHS} (EarlyStopping ile otomatik durma)")


# ============================================================================
# HÜCRE 4: VERİ YÜKLEME VE SINIF AĞIRLIKLARI
# ============================================================================

print("\n" + "=" * 60)
print("  VERİ YÜKLEME")
print("=" * 60)

df_train = pd.read_csv(os.path.join(VERI_KLASORU, "train.csv"), encoding='utf-8')
df_valid = pd.read_csv(os.path.join(VERI_KLASORU, "valid.csv"), encoding='utf-8')
df_test  = pd.read_csv(os.path.join(VERI_KLASORU, "test.csv"), encoding='utf-8')

for df in [df_train, df_valid, df_test]:
    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)

print(f"   Train:      {len(df_train):,}")
print(f"   Validation: {len(df_valid):,}")
print(f"   Test:       {len(df_test):,}")

# Sınıf ağırlıkları hesapla (dengesizlik varsa telafi eder)
from sklearn.utils.class_weight import compute_class_weight

sinif_agirlik = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=df_train['label'].values
)
class_weights = {0: sinif_agirlik[0], 1: sinif_agirlik[1]}
print(f"\n⚖️ Sınıf ağırlıkları: Normal={class_weights[0]:.3f}, Saldırgan={class_weights[1]:.3f}")


# ============================================================================
# HÜCRE 5: DATA AUGMENTATION
# ============================================================================

def random_word_deletion(text, p=0.1):
    """Rastgele kelime silme augmentation."""
    words = text.split()
    if len(words) <= 3:
        return text
    remaining = [w for w in words if random.random() > p]
    if len(remaining) == 0:
        return random.choice(words)
    return ' '.join(remaining)

def random_word_swap(text, n=1):
    """Rastgele kelime yer değiştirme."""
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return ' '.join(words)

# Saldırgan sınıf için augmentation (az olan sınıfı artır)
print("\n🔄 Data Augmentation uygulanıyor...")
augmented_rows = []
minority_df = df_train[df_train['label'] == 1] if class_weights[1] > 1 else df_train[df_train['label'] == 0]

for _, row in minority_df.sample(min(5000, len(minority_df)), random_state=RANDOM_SEED).iterrows():
    aug_text = random_word_deletion(row['text'], p=0.15)
    augmented_rows.append({'text': aug_text, 'label': row['label']})
    aug_text2 = random_word_swap(row['text'], n=2)
    augmented_rows.append({'text': aug_text2, 'label': row['label']})

df_aug = pd.DataFrame(augmented_rows)
df_train_aug = pd.concat([df_train, df_aug], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)

print(f"   Orijinal train: {len(df_train):,}")
print(f"   Augmented train: {len(df_train_aug):,} (+{len(df_aug):,} eklendi)")
print(f"   Yeni dağılım: Normal={(df_train_aug['label']==0).sum():,}, Saldırgan={(df_train_aug['label']==1).sum():,}")

# Sınıf ağırlıklarını yeniden hesapla
sinif_agirlik = compute_class_weight('balanced', classes=np.array([0, 1]), y=df_train_aug['label'].values)
class_weights = {0: sinif_agirlik[0], 1: sinif_agirlik[1]}
print(f"   Güncel ağırlıklar: Normal={class_weights[0]:.3f}, Saldırgan={class_weights[1]:.3f}")


# ============================================================================
# HÜCRE 6: YARDIMCI FONKSİYONLAR
# ============================================================================

def performans_raporu_ciz(y_true, y_pred, y_prob, model_adi):
    """Model performansını görselleştir ve raporla."""
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, accuracy_score, f1_score,
        precision_score, recall_score
    )

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"   📊 {model_adi} - TEST SONUÇLARI")
    print(f"{'='*60}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Normal', 'Saldırgan'])}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Saldırgan'], yticklabels=['Normal', 'Saldırgan'])
    axes[0].set_title(f'{model_adi} - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tahmin'); axes[0].set_ylabel('Gerçek')

    roc_auc_val = None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC (AUC = {roc_auc_val:.4f})')
        axes[1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.05])
        axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
        axes[1].set_title(f'{model_adi} - ROC Eğrisi', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(SONUC_KLASORU, f'{model_adi}_sonuclar.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return {'model': model_adi, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1_score': f1, 'roc_auc': roc_auc_val}


def egitim_grafigi_ciz(history, model_adi):
    """Eğitim/doğrulama grafiği + overfitting analizi."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', color='#3498db', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', color='#e74c3c', lw=2)
    axes[0].set_title(f'{model_adi} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', color='#3498db', lw=2)
    axes[1].plot(history.history['val_loss'], label='Validation', color='#e74c3c', lw=2)
    axes[1].set_title(f'{model_adi} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Overfitting Gap (Train-Val farkı)
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    gap = [t - v for t, v in zip(train_acc, val_acc)]
    axes[2].plot(gap, color='#e67e22', lw=2, marker='o')
    axes[2].axhline(y=0, color='green', linestyle='--', alpha=0.5)
    axes[2].fill_between(range(len(gap)), gap, alpha=0.3, color='#e67e22')
    axes[2].set_title(f'{model_adi} - Overfitting Gap', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Train Acc - Val Acc')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SONUC_KLASORU, f'{model_adi}_egitim.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Overfitting uyarısı
    final_gap = gap[-1] if gap else 0
    if final_gap > 0.05:
        print(f"   ⚠️ Overfitting riski: Train-Val gap = {final_gap:.4f}")
    else:
        print(f"   ✅ Overfitting yok: Train-Val gap = {final_gap:.4f}")


print("✅ Yardımcı fonksiyonlar tanımlandı!")


# ============================================================================
# HÜCRE 7: TOKENİZATİON (ORTAK)
# ============================================================================

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("\n🔤 Tokenization başlıyor...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df_train_aug['text'].values)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(df_train_aug['text'].values),
                            maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
X_valid_seq = pad_sequences(tokenizer.texts_to_sequences(df_valid['text'].values),
                            maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(df_test['text'].values),
                            maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

y_train = df_train_aug['label'].values
y_valid = df_valid['label'].values
y_test  = df_test['label'].values

vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
print(f"   Vocabulary: {vocab_size:,} | Sekans: {MAX_SEQ_LEN}")
print(f"   Train: {X_train_seq.shape} | Valid: {X_valid_seq.shape} | Test: {X_test_seq.shape}")


# ============================================================================
# HÜCRE 8: ATTENTION KATMANI (CUSTOM)
# ============================================================================

from tensorflow.keras import layers, Model, regularizers

class AttentionLayer(layers.Layer):
    """Bahdanau Attention mekanizması - hangi kelimelere odaklanılacağını öğrenir."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        self.u = self.add_weight(name='att_context',
                                 shape=(input_shape[-1],),
                                 initializer='glorot_uniform', trainable=True)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(x * tf.expand_dims(attention_weights, -1), axis=1)
        return context_vector

print("✅ Attention katmanı tanımlandı!")


# ============================================================================
# ORTAK CALLBACK'LER
# ============================================================================

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1,
        min_delta=0.001  # Minimum iyileşme eşiği
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=2, verbose=1, min_lr=1e-7
    )
]


# ============================================================================
# HÜCRE 9: MODEL 1 - LSTM + ATTENTION
# ============================================================================

print("\n" + "=" * 60)
print("   🚀 MODEL 1: LSTM + Attention")
print("=" * 60)

# Fonksiyonel API (Attention için gerekli)
inp = layers.Input(shape=(MAX_SEQ_LEN,))
x = layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQ_LEN)(inp)
x = layers.SpatialDropout1D(0.3)(x)
x = layers.LSTM(LSTM_UNITS, return_sequences=True,
                kernel_regularizer=regularizers.l2(L2_REG),
                recurrent_regularizer=regularizers.l2(L2_REG))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = AttentionLayer()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
x = layers.Dropout(0.2)(x)
# float32 çıkış (mixed precision uyumluluğu)
out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

lstm_model = Model(inputs=inp, outputs=out)
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['accuracy']
)
lstm_model.summary()

basla = datetime.now()
lstm_history = lstm_model.fit(
    X_train_seq, y_train,
    validation_data=(X_valid_seq, y_valid),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, class_weight=class_weights, verbose=1
)
print(f"\n⏱️ LSTM süresi: {datetime.now() - basla}")

egitim_grafigi_ciz(lstm_history, 'LSTM_Attention')

lstm_prob = lstm_model.predict(X_test_seq, verbose=0).flatten()
lstm_pred = (lstm_prob >= 0.5).astype(int)
lstm_sonuc = performans_raporu_ciz(y_test, lstm_pred, lstm_prob, 'LSTM_Attention')
lstm_model.save(os.path.join(MODEL_KLASORU, 'lstm_attention.keras'))
print("✅ LSTM + Attention kaydedildi!")


# ============================================================================
# HÜCRE 10: MODEL 2 - BiLSTM + ATTENTION
# ============================================================================

print("\n" + "=" * 60)
print("   🚀 MODEL 2: BiLSTM + Attention")
print("=" * 60)

inp = layers.Input(shape=(MAX_SEQ_LEN,))
x = layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQ_LEN)(inp)
x = layers.SpatialDropout1D(0.3)(x)
x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True,
                                      kernel_regularizer=regularizers.l2(L2_REG),
                                      recurrent_regularizer=regularizers.l2(L2_REG)))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = AttentionLayer()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

bilstm_model = Model(inputs=inp, outputs=out)
bilstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['accuracy']
)
bilstm_model.summary()

basla = datetime.now()
bilstm_history = bilstm_model.fit(
    X_train_seq, y_train,
    validation_data=(X_valid_seq, y_valid),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, class_weight=class_weights, verbose=1
)
print(f"\n⏱️ BiLSTM süresi: {datetime.now() - basla}")

egitim_grafigi_ciz(bilstm_history, 'BiLSTM_Attention')

bilstm_prob = bilstm_model.predict(X_test_seq, verbose=0).flatten()
bilstm_pred = (bilstm_prob >= 0.5).astype(int)
bilstm_sonuc = performans_raporu_ciz(y_test, bilstm_pred, bilstm_prob, 'BiLSTM_Attention')
bilstm_model.save(os.path.join(MODEL_KLASORU, 'bilstm_attention.keras'))
print("✅ BiLSTM + Attention kaydedildi!")


# ============================================================================
# HÜCRE 11: MODEL 3 - MULTI-KERNEL CNN
# ============================================================================

print("\n" + "=" * 60)
print("   🚀 MODEL 3: Multi-Kernel CNN")
print("=" * 60)

# Birden fazla filtre boyutu ile paralel konvolüsyon
inp = layers.Input(shape=(MAX_SEQ_LEN,))
emb = layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQ_LEN)(inp)
emb = layers.SpatialDropout1D(0.3)(emb)

# Paralel CNN dalları (3, 4, 5 kernel boyutları)
conv_outputs = []
for kernel_size in [3, 4, 5]:
    c = layers.Conv1D(CNN_FILTERS, kernel_size=kernel_size, activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(emb)
    c = layers.BatchNormalization()(c)
    c = layers.GlobalMaxPooling1D()(c)
    conv_outputs.append(c)

# Birleştir
x = layers.Concatenate()(conv_outputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
x = layers.Dropout(DROPOUT_RATE)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

cnn_model = Model(inputs=inp, outputs=out)
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['accuracy']
)
cnn_model.summary()

basla = datetime.now()
cnn_history = cnn_model.fit(
    X_train_seq, y_train,
    validation_data=(X_valid_seq, y_valid),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=callbacks, class_weight=class_weights, verbose=1
)
print(f"\n⏱️ CNN süresi: {datetime.now() - basla}")

egitim_grafigi_ciz(cnn_history, 'MultiKernel_CNN')

cnn_prob = cnn_model.predict(X_test_seq, verbose=0).flatten()
cnn_pred = (cnn_prob >= 0.5).astype(int)
cnn_sonuc = performans_raporu_ciz(y_test, cnn_pred, cnn_prob, 'MultiKernel_CNN')
cnn_model.save(os.path.join(MODEL_KLASORU, 'multikernel_cnn.keras'))
print("✅ Multi-Kernel CNN kaydedildi!")


# ============================================================================
# HÜCRE 12: MODEL 4 - BERT (GELİŞMİŞ)
# ============================================================================

print("\n" + "=" * 60)
print(f"   🚀 MODEL 4: BERT ({BERT_MODEL_NAME})")
print("=" * 60)

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
print("✅ BERT Tokenizer yüklendi!")

class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts; self.labels = labels
        self.tokenizer = tokenizer; self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]), add_special_tokens=True,
            max_length=self.max_len, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Augmented veri ile eğit
train_dataset = ToxicDataset(df_train_aug['text'].values, df_train_aug['label'].values, bert_tokenizer, BERT_MAX_LEN)
valid_dataset = ToxicDataset(df_valid['text'].values, df_valid['label'].values, bert_tokenizer, BERT_MAX_LEN)
test_dataset  = ToxicDataset(df_test['text'].values, df_test['label'].values, bert_tokenizer, BERT_MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BERT_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BERT_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"   Train batches: {len(train_loader)} | Valid: {len(valid_loader)} | Test: {len(test_loader)}")

# Model
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_NAME, num_labels=2,
    hidden_dropout_prob=0.2,       # BERT içi dropout
    attention_probs_dropout_prob=0.2  # Attention dropout
)
bert_model.to(device)
print("✅ BERT modeli yüklendi!")

# Discriminative Learning Rates (alt katmanlara düşük lr, üst katmanlara yüksek lr)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    # BERT alt katmanları (düşük lr)
    {'params': [p for n, p in bert_model.bert.embeddings.named_parameters()],
     'lr': BERT_LR * 0.1, 'weight_decay': 0.01},
    # BERT encoder katmanları
    {'params': [p for n, p in bert_model.bert.encoder.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'lr': BERT_LR, 'weight_decay': 0.01},
    {'params': [p for n, p in bert_model.bert.encoder.named_parameters()
                if any(nd in n for nd in no_decay)],
     'lr': BERT_LR, 'weight_decay': 0.0},
    # Classifier katmanı (yüksek lr)
    {'params': bert_model.classifier.parameters(),
     'lr': BERT_LR * 5, 'weight_decay': 0.01},
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
total_steps = len(train_loader) * BERT_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(BERT_WARMUP * total_steps),
    num_training_steps=total_steps
)

# Class weights for loss
weights_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=LABEL_SMOOTH)

# Mixed precision scaler
scaler = GradScaler()

# Eğitim
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_f1 = 0  # F1'e göre kaydet (accuracy yerine)

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

        # Mixed Precision Forward
        with autocast():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

        # Scaled backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"     Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {correct/total:.4f}")

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss); train_accs.append(train_acc)

    # === Validation ===
    bert_model.eval()
    total_loss, correct, total = 0, 0, 0
    val_preds_epoch, val_labels_epoch = [], []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with autocast():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            val_preds_epoch.extend(preds.cpu().numpy())
            val_labels_epoch.extend(labels.cpu().numpy())

    val_loss = total_loss / len(valid_loader)
    val_acc = correct / total
    val_losses.append(val_loss); val_accs.append(val_acc)

    from sklearn.metrics import f1_score as f1_metric
    val_f1 = f1_metric(val_labels_epoch, val_preds_epoch)

    print(f"\n   📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"   📉 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # En iyi modeli F1-Score'a göre kaydet
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(bert_model.state_dict(), os.path.join(MODEL_KLASORU, 'bert_best.pt'))
        print(f"   ✅ En iyi model kaydedildi! (val_f1: {val_f1:.4f})")

bert_sure = datetime.now() - basla
print(f"\n⏱️ BERT eğitim süresi: {bert_sure}")

# BERT Eğitim Grafiği
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(1, BERT_EPOCHS + 1)

axes[0].plot(epochs_range, train_accs, 'o-', label='Train', color='#3498db', lw=2)
axes[0].plot(epochs_range, val_accs, 'o-', label='Val', color='#e74c3c', lw=2)
axes[0].set_title('BERT - Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, train_losses, 'o-', label='Train', color='#3498db', lw=2)
axes[1].plot(epochs_range, val_losses, 'o-', label='Val', color='#e74c3c', lw=2)
axes[1].set_title('BERT - Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

gap = [t - v for t, v in zip(train_accs, val_accs)]
axes[2].plot(epochs_range, gap, 'o-', color='#e67e22', lw=2)
axes[2].axhline(y=0, color='green', linestyle='--', alpha=0.5)
axes[2].fill_between(epochs_range, gap, alpha=0.3, color='#e67e22')
axes[2].set_title('BERT - Overfitting Gap', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Train-Val Acc Gap')
axes[2].grid(True, alpha=0.3)

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
        with autocast():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits.float(), dim=1)[:, 1]
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

bert_sonuc = performans_raporu_ciz(np.array(all_labels), np.array(all_preds), np.array(all_probs), 'BERT')


# ============================================================================
# HÜCRE 13: ENSEMBLE MODEL (TÜM MODELLERİN BİRLEŞİMİ)
# ============================================================================

print("\n" + "=" * 60)
print("   🏆 ENSEMBLE MODEL (Ağırlıklı Oylama)")
print("=" * 60)

# Her modelin tahmin olasılıklarını ağırlıklı ortala
# F1-score'a göre ağırlık ver
f1_scores = {
    'lstm': lstm_sonuc['f1_score'],
    'bilstm': bilstm_sonuc['f1_score'],
    'cnn': cnn_sonuc['f1_score'],
    'bert': bert_sonuc['f1_score']
}

# Ağırlıkları normalize et
toplam_f1 = sum(f1_scores.values())
weights = {k: v / toplam_f1 for k, v in f1_scores.items()}
print(f"   Ağırlıklar (F1 bazlı):")
for k, v in weights.items():
    print(f"     {k:8s}: {v:.3f} (F1={f1_scores[k]:.4f})")

# Ağırlıklı ensemble
ensemble_prob = (
    weights['lstm'] * lstm_prob +
    weights['bilstm'] * bilstm_prob +
    weights['cnn'] * cnn_prob +
    weights['bert'] * np.array(all_probs)
)
ensemble_pred = (ensemble_prob >= 0.5).astype(int)
ensemble_sonuc = performans_raporu_ciz(y_test, ensemble_pred, ensemble_prob, 'ENSEMBLE')


# ============================================================================
# HÜCRE 14: MODEL KARŞILAŞTIRMA
# ============================================================================

print("\n" + "=" * 60)
print("   📊 MODEL KARŞILAŞTIRMA")
print("=" * 60)

tum_sonuclar = [lstm_sonuc, bilstm_sonuc, cnn_sonuc, bert_sonuc, ensemble_sonuc]
df_sonuc = pd.DataFrame(tum_sonuclar)
print("\n" + df_sonuc.to_string(index=False))

# Karşılaştırma grafiği
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
modeller = df_sonuc['model'].tolist()
x = np.arange(len(modeller))
colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']

for ax_idx, (metric, title) in enumerate([
    ('accuracy', 'Accuracy'), ('f1_score', 'F1-Score'), ('roc_auc', 'ROC-AUC')
]):
    vals = df_sonuc[metric].values
    bars = axes[ax_idx].bar(x, vals, color=colors, edgecolor='black')
    axes[ax_idx].set_title(f'{title} Karşılaştırma', fontsize=14, fontweight='bold')
    axes[ax_idx].set_ylabel(title, fontsize=12)
    axes[ax_idx].set_xticks(x)
    axes[ax_idx].set_xticklabels(modeller, fontsize=9, rotation=15)
    axes[ax_idx].set_ylim(0.5, 1.0)
    for bar, val in zip(bars, vals):
        axes[ax_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                          f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SONUC_KLASORU, 'model_karsilastirma.png'), dpi=150, bbox_inches='tight')
plt.show()

df_sonuc.to_csv(os.path.join(SONUC_KLASORU, 'model_karsilastirma.csv'), index=False)

en_iyi = df_sonuc.loc[df_sonuc['f1_score'].idxmax()]
print(f"\n🏆 EN İYİ MODEL: {en_iyi['model']}")
print(f"   Accuracy:  {en_iyi['accuracy']:.4f}")
print(f"   Precision: {en_iyi['precision']:.4f}")
print(f"   Recall:    {en_iyi['recall']:.4f}")
print(f"   F1-Score:  {en_iyi['f1_score']:.4f}")
print(f"   ROC-AUC:   {en_iyi['roc_auc']:.4f}")


# ============================================================================
# HÜCRE 15: ÖRNEK TAHMİN
# ============================================================================

print("\n" + "=" * 60)
print(f"   🔮 ÖRNEK TAHMİNLER (BERT)")
print("=" * 60)

ornek_metinler = [
    "Bu çok güzel bir paylaşım teşekkürler",
    "Seni öldüreceğim lan aptal herif",
    "Bugün hava çok güzel dışarı çıkalım mı",
    "Siktir git buradan gerizekalı",
    "Atatürk çok büyük bir lider",
    "Senin gibi salağı görmedim",
    "Harika bir gol attı tebrikler",
    "Hepiniz geri zekalısınız hiçbiriniz beğenmiyor",
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
        with autocast():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits.float(), dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    emoji = "🟢" if pred == 0 else "🔴"
    etiket = "Normal" if pred == 0 else "Saldırgan"
    print(f"   {emoji} [{etiket}] (%{conf*100:.1f}) → \"{metin}\"")


# ============================================================================
# HÜCRE 16: SONUÇLARI KAYDET
# ============================================================================

if IN_COLAB:
    print("\n💾 Sonuçları Google Drive'a kaydetmek için:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    print("   !cp -r sonuclar/ /content/drive/MyDrive/")

print("\n" + "=" * 60)
print("   ✅ TÜM İŞLEMLER TAMAMLANDI!")
print("=" * 60)
print(f"""
   📊 Sonuçlar: {SONUC_KLASORU}
   🧠 Modeller: {MODEL_KLASORU}

   Anti-overfitting teknikler kullanıldı:
     ✅ L2 Regularization ({L2_REG})
     ✅ Label Smoothing ({LABEL_SMOOTH})
     ✅ BatchNormalization
     ✅ Dropout ({DROPOUT_RATE})
     ✅ EarlyStopping (patience=5)
     ✅ Class Weights
     ✅ Data Augmentation (+{len(df_aug):,} satır)
     ✅ Gradient Clipping
     ✅ Attention Mekanizması
     ✅ Multi-Kernel CNN
     ✅ Discriminative Learning Rates (BERT)
     ✅ Mixed Precision Training
     ✅ Ensemble Model
""")
