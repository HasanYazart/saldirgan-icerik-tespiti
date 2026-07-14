# -*- coding: utf-8 -*-
"""Belge (3) uyumlu Word2Vec + CNN/LSTM ve BERT eğitim pipeline'ı.

Öne çıkan korumalar:
  * tokenizer/vectorizer yalnızca train ile öğrenilir;
  * model ve eşik seçimi yalnızca validation üzerinde yapılır;
  * test split'e eğitim, kalibrasyon veya model seçiminde dokunulmaz;
  * Word2Vec yalnızca train metinleriyle öğrenilir;
  * belgede önerilen CNN -> LSTM hibrit mimarisi uygulanır;
  * Keras'ta token dropout, AdamW, L2, label smoothing ve early stopping;
  * Transformer'da focal loss, AdamW, cosine schedule, warmup, gradient
    checkpointing, mixed precision ve desteklenen GPU'larda torch.compile;
  * olasılık kalibrasyonu ve hedef yanlış-pozitif oranına bağlı eşik.

Colab:
    !pip install -e ".[colab]"
    !python veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py --prepare-data --models lstm,cnn,cnn_lstm
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend_api.text_processing import PREPROCESSING_VERSION as BACKEND_PREPROCESSING_VERSION
from veri_temizlemesi_ve_egitimi.document_preprocessing import (
    PREPROCESSING_VERSION,
    normalize_for_document,
)


try:
    from transformers import Trainer as _TrainerBase
except ImportError:  # Import-safe dry-run ve yardımcı fonksiyon testleri için.
    class _TrainerBase:  # type: ignore[no-redef]
        pass


def set_global_seed(
    seed: int,
    *,
    seed_torch: bool = True,
    seed_tensorflow: bool = True,
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except ImportError:
            pass
    if seed_tensorflow:
        try:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(seed)
            try:
                tf.config.experimental.enable_op_determinism()
            except (AttributeError, RuntimeError):
                pass
        except ImportError:
            pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        column for column in ("text", "label", "kaynak", "group_id") if column in frame
    ]
    payload = frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_token_hash(token: str) -> int:
    """Word2Vec başlangıç vektörlerini süreçler arasında tekrarlanabilir yapar."""

    return int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little"
    )


@dataclass(slots=True)
class TrainingConfig:
    project_dir: Path = PROJECT_DIR
    data_dir: Path | None = None
    result_dir: Path | None = None
    model_dir: Path | None = None
    models: tuple[str, ...] = ("lstm", "cnn", "cnn_lstm")
    seed: int = 42
    target_false_positive_rate: float = 0.01
    deploy_backend: bool = False

    # Keras
    max_tokens: int = 50_000
    sequence_length: int = 192
    embedding_dim: int = 192
    word2vec_window: int = 5
    word2vec_min_count: int = 2
    word2vec_epochs: int = 10
    recurrent_units: int = 96
    cnn_filters: int = 192
    dropout: float = 0.45
    token_dropout: float = 0.08
    l2_regularization: float = 1e-4
    label_smoothing: float = 0.05
    keras_learning_rate: float = 3e-4
    keras_batch_size: int = 64
    keras_epochs: int = 30
    keras_patience: int = 5

    # Transformer
    transformer_model: str = "dbmdz/bert-base-turkish-cased"
    transformer_max_length: int = 160
    transformer_batch_size: int = 16
    transformer_eval_batch_size: int = 32
    transformer_epochs: float = 3.0
    transformer_learning_rate: float = 2e-5
    transformer_weight_decay: float = 0.02
    transformer_warmup_ratio: float = 0.10
    transformer_gradient_accumulation: int = 2
    transformer_patience: int = 2
    focal_gamma: float = 2.0
    # Colab'in sik kullandigi T4 ortaminda derleme ilk calistirmayi ciddi
    # uzatabildigi ve surum uyumsuzluklarina acik oldugu icin opt-in'dir.
    torch_compile: bool = False
    gradient_checkpointing: bool = True
    resume_from_checkpoint: Path | str | None = None

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).resolve()
        self.data_dir = Path(self.data_dir or self.project_dir / "veri_setleri").resolve()
        self.result_dir = Path(self.result_dir or self.project_dir / "sonuclar").resolve()
        self.model_dir = Path(self.model_dir or self.project_dir / "modeller").resolve()
        if self.resume_from_checkpoint not in (None, "auto"):
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint).resolve()
        self.models = tuple(name.strip().lower() for name in self.models if name.strip())
        unknown = set(self.models) - {"lstm", "cnn", "cnn_lstm", "bert"}
        if unknown:
            raise ValueError(f"Bilinmeyen modeller: {sorted(unknown)}")
        if not 0 < self.target_false_positive_rate < 1:
            raise ValueError("target_false_positive_rate 0-1 arasında olmalı")
        if self.word2vec_epochs < 1 or self.word2vec_min_count < 1:
            raise ValueError("Word2Vec epochs ve min_count en az 1 olmalı")
        if not self.models:
            raise ValueError("En az bir model seçilmeli")


class TrainingDataModule:
    """Split'leri yükler ve eğitim başlamadan veri sızıntısını reddeder."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.frames: dict[str, pd.DataFrame] = {}
        self.dataset_manifest: dict[str, Any] = {}

    def load(self) -> "TrainingDataModule":
        manifest_path = self.config.data_dir / "dataset_manifest.json"
        has_manifest = manifest_path.exists()
        if has_manifest:
            self.dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = self.dataset_manifest.get("preprocessing_version")
            if expected != PREPROCESSING_VERSION:
                raise ValueError(
                    f"Ön işleme sürümü uyuşmuyor: veri={expected}, kod={PREPROCESSING_VERSION}"
                )

        for split in ("train", "valid", "test"):
            path = self.config.data_dir / f"{split}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} bulunamadı. Önce 01_veri_temizleme.py çalıştırın."
                )
            # group_id yalnizca rakamlardan olusabildiginde otomatik sayisal
            # tur cikarimi bastaki sifirlari silebilir ve manifest hash'ini
            # bozabilir. Etiket asagida kontrollu olarak yeniden sayiya cevrilir.
            frame = pd.read_csv(path, dtype=str)
            missing = {"text", "label"} - set(frame.columns)
            if missing:
                raise ValueError(f"{path}: eksik sütunlar {sorted(missing)}")
            frame = frame.copy()
            if has_manifest:
                # Manifestli split temizleme pipeline'inda zaten tam bir kez
                # normalize edilmistir. Leetspeak cozumunden sonra yeni bir
                # hashtag deseni olusabildigi icin ikinci normalizasyon her
                # metinde idempotent degildir ve veri parmak izini bozar.
                frame["text"] = frame["text"].fillna("").astype(str)
            else:
                # Eski, manifestsiz splitler icin geriye uyumluluk.
                frame["text"] = frame["text"].map(normalize_for_document)
            frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int8")
            if not set(frame["label"].unique()).issubset({0, 1}):
                raise ValueError(f"{path}: etiketler yalnızca 0/1 olabilir")
            if frame["text"].eq("").any():
                raise ValueError(f"{path}: boş metin bulundu")
            if frame["text"].duplicated().any():
                raise ValueError(f"{path}: normalize edilmiş yinelenen metin bulundu")
            self.frames[split] = frame.reset_index(drop=True)

        if has_manifest:
            expected_fingerprints = self.dataset_manifest.get("split_fingerprints", {})
            for split, frame in self.frames.items():
                expected_fingerprint = expected_fingerprints.get(split)
                if expected_fingerprint and frame_fingerprint(frame) != expected_fingerprint:
                    raise ValueError(
                        f"{split}.csv parmak izi manifestle uyuşmuyor; veri değiştirilmiş olabilir"
                    )
        self._assert_no_leakage()
        self._print_summary()
        return self

    def _assert_no_leakage(self) -> None:
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            text_overlap = set(self.frames[left]["text"]) & set(self.frames[right]["text"])
            if text_overlap:
                raise ValueError(f"{left}-{right} arasında {len(text_overlap)} aynı metin var")
            if "group_id" in self.frames[left] and "group_id" in self.frames[right]:
                group_overlap = set(self.frames[left]["group_id"]) & set(
                    self.frames[right]["group_id"]
                )
                if group_overlap:
                    raise ValueError(
                        f"{left}-{right} arasında {len(group_overlap)} yakın-kopya grubu var"
                    )

    def _print_summary(self) -> None:
        print("\nVeri özeti")
        for name, frame in self.frames.items():
            distribution = frame["label"].value_counts().sort_index().to_dict()
            print(f"  {name:5s}: {len(frame):,} | sınıflar={distribution}")

    @property
    def train(self) -> pd.DataFrame:
        return self.frames["train"]

    @property
    def valid(self) -> pd.DataFrame:
        return self.frames["valid"]

    @property
    def test(self) -> pd.DataFrame:
        return self.frames["test"]

    def class_weights(self) -> dict[int, float]:
        labels = self.train["label"].to_numpy()
        values = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
        # Aşırı ağırlıklar kararsız eğitime neden olmasın.
        values = np.clip(values, 0.25, 4.0)
        return {0: float(values[0]), 1: float(values[1])}


class TemperatureScaler:
    """Validation NLL ile tek parametreli olasılık kalibrasyonu."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.log(probabilities / (1 - probabilities))

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        labels = np.asarray(labels, dtype=np.float64)
        if len(np.unique(labels)) < 2:
            self.temperature = 1.0
            return self
        logits = self._logit(probabilities)
        candidates = np.exp(np.linspace(math.log(0.35), math.log(4.0), 240))
        best = (float("inf"), 1.0)
        for temperature in candidates:
            calibrated = 1 / (1 + np.exp(-np.clip(logits / temperature, -40, 40)))
            nll = -np.mean(
                labels * np.log(np.clip(calibrated, 1e-8, 1))
                + (1 - labels) * np.log(np.clip(1 - calibrated, 1e-8, 1))
            )
            if nll < best[0]:
                best = (float(nll), float(temperature))
        self.temperature = best[1]
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = self._logit(probabilities) / self.temperature
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


class ThresholdOptimizer:
    """Validation'da FPR bütçesini aşmadan en yüksek F1 eşiğini seçer."""

    def __init__(self, target_fpr: float) -> None:
        self.target_fpr = target_fpr

    def select(self, probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
        probabilities = np.asarray(probabilities)
        labels = np.asarray(labels).astype(int)
        candidates = np.unique(
            np.concatenate([np.linspace(0.50, 0.999, 750), np.quantile(probabilities, np.linspace(0, 1, 250))])
        )
        best: tuple[tuple[float, float, float, float], float, dict[str, float]] | None = None
        for threshold in candidates:
            if not 0.50 <= threshold <= 0.999:
                continue
            predictions = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            fpr = fp / max(fp + tn, 1)
            if fpr > self.target_fpr:
                continue
            f1 = f1_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            precision = precision_score(labels, predictions, zero_division=0)
            rank = (f1, recall, precision, -float(threshold))
            details = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "false_positive_rate": float(fpr),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
            if best is None or rank > best[0]:
                best = (rank, float(threshold), details)
        if best is None:
            threshold = 0.999
            predictions = (probabilities >= threshold).astype(int)
            tn, fp, fn, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            return threshold, {
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "false_positive_rate": float(fp / max(fp + tn, 1)),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        return best[1], best[2]


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    boundaries = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (probabilities > lower) & (probabilities <= upper)
        if not mask.any():
            continue
        error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    per_class_precision = precision_score(
        labels, predictions, labels=[0, 1], average=None, zero_division=0
    )
    per_class_recall = recall_score(
        labels, predictions, labels=[0, 1], average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        labels, predictions, labels=[0, 1], average=None, zero_division=0
    )
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "class_0_precision": float(per_class_precision[0]),
        "class_0_recall": float(per_class_recall[0]),
        "class_0_f1": float(per_class_f1[0]),
        "class_1_precision": float(per_class_precision[1]),
        "class_1_recall": float(per_class_recall[1]),
        "class_1_f1": float(per_class_f1[1]),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


@dataclass(slots=True)
class PredictionBundle:
    model_name: str
    temperature: float
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    artifact: str


class BaseModelTrainer:
    def __init__(self, config: TrainingConfig, data: TrainingDataModule) -> None:
        self.config = config
        self.data = data

    def _calibrate_and_evaluate(
        self,
        model_name: str,
        validation_probabilities: np.ndarray,
        test_probabilities: np.ndarray,
        artifact: Path,
    ) -> PredictionBundle:
        validation_labels = self.data.valid["label"].to_numpy()
        test_labels = self.data.test["label"].to_numpy()
        scaler = TemperatureScaler().fit(validation_probabilities, validation_labels)
        calibrated_validation = scaler.transform(validation_probabilities)
        calibrated_test = scaler.transform(test_probabilities)
        threshold, _ = ThresholdOptimizer(self.config.target_false_positive_rate).select(
            calibrated_validation, validation_labels
        )
        return PredictionBundle(
            model_name=model_name,
            temperature=scaler.temperature,
            threshold=threshold,
            validation_metrics=calculate_metrics(validation_labels, calibrated_validation, threshold),
            test_metrics=calculate_metrics(test_labels, calibrated_test, threshold),
            artifact=str(artifact),
        )


@dataclass(slots=True)
class KerasTextResources:
    vectorizer: Any
    embedding_matrix: np.ndarray


class KerasModelTrainer(BaseModelTrainer):
    """Belgedeki Word2Vec, LSTM, CNN ve CNN->LSTM modellerini eğitir."""

    def __init__(
        self,
        config: TrainingConfig,
        data: TrainingDataModule,
        model_name: str,
        resource_cache: dict[str, KerasTextResources],
    ) -> None:
        super().__init__(config, data)
        self.model_name = model_name
        self.resource_cache = resource_cache

    def _build_text_resources(self, tf: Any) -> KerasTextResources:
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise RuntimeError(
                "Belge (3) Word2Vec adımı için gensim kurun: pip install -e '.[training]'"
            ) from exc

        sentences = [
            text.split()
            for text in self.data.train["text"].astype(str)
            if str(text).strip()
        ]
        print(f"Word2Vec yalnızca train üzerinde eğitiliyor: {len(sentences):,} metin")
        word2vec = Word2Vec(
            sentences=sentences,
            vector_size=self.config.embedding_dim,
            window=self.config.word2vec_window,
            min_count=self.config.word2vec_min_count,
            workers=1,
            sg=1,
            negative=10,
            epochs=self.config.word2vec_epochs,
            seed=self.config.seed,
            hashfxn=stable_token_hash,
            max_final_vocab=max(self.config.max_tokens - 2, 100),
        )
        vocabulary = word2vec.wv.index_to_key[: self.config.max_tokens - 2]
        vectorizer = tf.keras.layers.TextVectorization(
            vocabulary=vocabulary,
            output_mode="int",
            output_sequence_length=self.config.sequence_length,
            standardize=None,
            split="whitespace",
            name="train_only_word2vec_vectorizer",
        )
        keras_vocabulary = vectorizer.get_vocabulary()
        embedding_matrix = np.zeros(
            (len(keras_vocabulary), self.config.embedding_dim), dtype=np.float32
        )
        if vocabulary:
            embedding_matrix[1] = np.mean(
                word2vec.wv.vectors[: len(vocabulary)], axis=0
            )
        for index, token in enumerate(keras_vocabulary[2:], start=2):
            embedding_matrix[index] = word2vec.wv[token]

        metadata = {
            "method": "Word2Vec skip-gram",
            "trained_on": "train_only",
            "vocabulary_size": len(keras_vocabulary),
            "vector_size": self.config.embedding_dim,
            "window": self.config.word2vec_window,
            "min_count": self.config.word2vec_min_count,
            "epochs": self.config.word2vec_epochs,
            "workers": 1,
            "seed": self.config.seed,
        }
        (self.config.result_dir / "word2vec_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        del word2vec
        gc.collect()
        return KerasTextResources(vectorizer, embedding_matrix)

    def _resources(self, tf: Any) -> KerasTextResources:
        if "document_word2vec" not in self.resource_cache:
            self.resource_cache["document_word2vec"] = self._build_text_resources(tf)
        return self.resource_cache["document_word2vec"]

    def _build_model(self, tf: Any, resources: KerasTextResources) -> Any:
        layers = tf.keras.layers
        regularizers = tf.keras.regularizers
        config = self.config

        @tf.keras.utils.register_keras_serializable(package="TurkishToxic")
        class TokenDropout(layers.Layer):
            def __init__(self, rate: float, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.rate = rate

            def call(self, inputs: Any, training: bool | None = None) -> Any:
                if self.rate <= 0 or training is None:
                    return inputs

                def apply_dropout() -> Any:
                    random_values = tf.random.uniform(tf.shape(inputs))
                    drop_mask = tf.logical_and(random_values < self.rate, inputs > 1)
                    return tf.where(drop_mask, tf.zeros_like(inputs), inputs)

                if isinstance(training, bool):
                    return apply_dropout() if training else inputs
                return tf.cond(tf.cast(training, tf.bool), apply_dropout, lambda: inputs)

            def get_config(self) -> dict[str, Any]:
                return {**super().get_config(), "rate": self.rate}

        text_input = layers.Input(shape=(), dtype=tf.string, name="text")
        token_ids = resources.vectorizer(text_input)
        token_ids = TokenDropout(config.token_dropout, name="token_dropout")(token_ids)
        embedding = layers.Embedding(
            input_dim=resources.embedding_matrix.shape[0],
            output_dim=config.embedding_dim,
            weights=[resources.embedding_matrix],
            trainable=False,
            mask_zero=False,
            name="train_only_word2vec_embedding",
        )(token_ids)
        x = layers.SpatialDropout1D(config.dropout * 0.65)(embedding)
        regularizer = regularizers.l2(config.l2_regularization)

        if self.model_name == "lstm":
            x = layers.LSTM(
                config.recurrent_units,
                return_sequences=True,
                dropout=config.dropout * 0.45,
                recurrent_dropout=0.0,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                name="lstm_encoder",
            )(x)
            x = layers.Concatenate()(
                [layers.GlobalMaxPooling1D()(x), layers.GlobalAveragePooling1D()(x)]
            )
        elif self.model_name == "cnn":
            branches = []
            for kernel_size in (2, 3, 4, 5):
                branch = layers.Conv1D(
                    config.cnn_filters,
                    kernel_size,
                    padding="same",
                    activation="swish",
                    kernel_regularizer=regularizer,
                )(x)
                branch = layers.GlobalMaxPooling1D()(branch)
                branches.append(branch)
            x = layers.Concatenate(name="multi_kernel_features")(branches)
        else:  # Belge (3)'te önerilen CNN -> RNN/LSTM hibrit modeli.
            x = layers.Conv1D(
                config.cnn_filters,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizer,
                name="local_ngram_cnn",
            )(x)
            x = layers.MaxPooling1D(pool_size=2, name="cnn_max_pooling")(x)
            x = layers.LSTM(
                config.recurrent_units,
                return_sequences=True,
                dropout=config.dropout * 0.45,
                recurrent_dropout=0.0,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                name="long_term_lstm",
            )(x)
            x = layers.Concatenate(name="hybrid_features")(
                [layers.GlobalMaxPooling1D()(x), layers.GlobalAveragePooling1D()(x)]
            )

        x = layers.Dense(128, activation="swish", kernel_regularizer=regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout)(x)
        output = layers.Dense(1, activation="sigmoid", dtype="float32", name="toxicity")(x)
        model = tf.keras.Model(text_input, output, name=f"turkish_toxic_{self.model_name}")
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.keras_learning_rate,
            weight_decay=config.l2_regularization,
            global_clipnorm=1.0,
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.label_smoothing),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            ],
        )
        return model

    @staticmethod
    def _dataset(tf: Any, frame: pd.DataFrame, batch_size: int, training: bool, seed: int) -> Any:
        dataset = tf.data.Dataset.from_tensor_slices(
            (frame["text"].astype(str).to_numpy(), frame["label"].astype("float32").to_numpy())
        )
        if training:
            dataset = dataset.shuffle(min(len(frame), 50_000), seed=seed, reshuffle_each_iteration=True)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def _save_history(self, history: Any) -> None:
        values = {
            key: [float(value) for value in series]
            for key, series in history.history.items()
        }
        (self.config.result_dir / f"{self.model_name}_history.json").write_text(
            json.dumps(values, indent=2), encoding="utf-8"
        )

        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(values.get("loss", []), label="Train")
        axes[0].plot(values.get("val_loss", []), label="Validation")
        axes[0].set_title(f"{self.model_name.upper()} kayıp")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(values.get("accuracy", []), label="Train")
        axes[1].plot(values.get("val_accuracy", []), label="Validation")
        axes[1].set_title(f"{self.model_name.upper()} doğruluk")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(
            self.config.result_dir / f"{self.model_name}_egitim.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

    def train(self) -> PredictionBundle:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("Keras modelleri için TensorFlow kurun: pip install -e '.[training]'") from exc

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        if gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        tf.keras.utils.set_random_seed(self.config.seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass

        resources = self._resources(tf)
        model = self._build_model(tf, resources)
        model.summary()

        artifact = self.config.model_dir / f"{self.model_name}_best.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                artifact, monitor="val_loss", mode="min", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.config.keras_patience,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        history = model.fit(
            self._dataset(tf, self.data.train, self.config.keras_batch_size, True, self.config.seed),
            validation_data=self._dataset(
                tf, self.data.valid, self.config.keras_batch_size, False, self.config.seed
            ),
            epochs=self.config.keras_epochs,
            callbacks=callbacks,
            class_weight=self.data.class_weights(),
            verbose=2,
        )
        self._save_history(history)
        validation_probs = model.predict(
            self._dataset(tf, self.data.valid, self.config.keras_batch_size, False, self.config.seed),
            verbose=0,
        ).reshape(-1)
        test_probs = model.predict(
            self._dataset(tf, self.data.test, self.config.keras_batch_size, False, self.config.seed),
            verbose=0,
        ).reshape(-1)
        bundle = self._calibrate_and_evaluate(
            self.model_name, validation_probs, test_probs, artifact
        )
        tf.keras.backend.clear_session()
        gc.collect()
        return bundle


class FocalLossTrainer(_TrainerBase):
    """HuggingFace Trainer üzerinde class-weighted focal loss."""

    def __init__(
        self,
        *args: Any,
        focal_gamma: float = 2.0,
        class_weights: Sequence[float] = (1.0, 1.0),
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.focal_gamma = float(focal_gamma)
        self.focal_class_weights = tuple(float(value) for value in class_weights)
        self.focal_label_smoothing = float(label_smoothing)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        logits = outputs.logits
        weights = torch.as_tensor(
            self.focal_class_weights, dtype=logits.dtype, device=logits.device
        )
        cross_entropy = functional.cross_entropy(
            logits,
            labels,
            weight=weights,
            reduction="none",
            label_smoothing=self.focal_label_smoothing,
        )
        true_class_probability = torch.softmax(logits.float(), dim=-1).gather(
            1, labels.unsqueeze(1)
        ).squeeze(1)
        loss = ((1 - true_class_probability) ** self.focal_gamma * cross_entropy).mean()
        return (loss, outputs) if return_outputs else loss


class TransformerModelTrainer(BaseModelTrainer):
    class TextDataset:
        def __init__(self, encodings: dict[str, Any], labels: np.ndarray) -> None:
            self.encodings = encodings
            self.labels = labels.astype(np.int64)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            import torch

            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    def _tokenize(self, tokenizer: Any, frame: pd.DataFrame) -> "TransformerModelTrainer.TextDataset":
        encodings = tokenizer(
            frame["text"].astype(str).tolist(),
            truncation=True,
            max_length=self.config.transformer_max_length,
            padding=False,
        )
        return self.TextDataset(encodings, frame["label"].to_numpy())

    @staticmethod
    def _trainer_metrics(evaluation: Any) -> dict[str, float]:
        logits, labels = evaluation
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits)
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)[:, 1] / np.exp(logits).sum(axis=1)
        return calculate_metrics(np.asarray(labels), probabilities, 0.5)

    @staticmethod
    def _probabilities(prediction_output: Any) -> np.ndarray:
        logits = prediction_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials[:, 1] / exponentials.sum(axis=1)

    def train(self) -> PredictionBundle:
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
                DataCollatorWithPadding,
                EarlyStoppingCallback,
                TrainingArguments,
            )
            from transformers.trainer_utils import get_last_checkpoint
        except ImportError as exc:
            raise RuntimeError("BERT için torch ve transformers kurun: pip install -e '.[ml]'") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.config.transformer_model, use_fast=True)
        model_config = AutoConfig.from_pretrained(
            self.config.transformer_model,
            num_labels=2,
            id2label={0: "clean", 1: "toxic"},
            label2id={"clean": 0, "toxic": 1},
        )
        for attribute in ("hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout"):
            if hasattr(model_config, attribute):
                setattr(model_config, attribute, min(self.config.dropout * 0.5, 0.30))
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.transformer_model, config=model_config
        )
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        train_dataset = self._tokenize(tokenizer, self.data.train)
        valid_dataset = self._tokenize(tokenizer, self.data.valid)
        test_dataset = self._tokenize(tokenizer, self.data.test)
        cuda = torch.cuda.is_available()
        bf16 = bool(cuda and torch.cuda.is_bf16_supported())
        tf32 = bool(cuda and torch.cuda.get_device_capability()[0] >= 8)
        if cuda:
            print(f"GPU: {torch.cuda.get_device_name(0)} | bf16={bf16} | tf32={tf32}")
        else:
            print("UYARI: CUDA GPU bulunamadi; BERT egitimi CPU'da cok uzun surebilir.")
        compile_enabled = bool(
            self.config.torch_compile
            and cuda
            and platform.system().lower() != "windows"
            and hasattr(torch, "compile")
        )
        if self.config.torch_compile and not compile_enabled:
            print("torch.compile bu ortamda güvenli/destekli değil; eager moda geçildi.")

        run_dir = self.config.model_dir / "bert_checkpoints"
        steps_per_epoch = math.ceil(
            len(self.data.train)
            / (self.config.transformer_batch_size * self.config.transformer_gradient_accumulation)
        )
        arguments = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=self.config.transformer_epochs,
            per_device_train_batch_size=self.config.transformer_batch_size,
            per_device_eval_batch_size=self.config.transformer_eval_batch_size,
            gradient_accumulation_steps=self.config.transformer_gradient_accumulation,
            learning_rate=self.config.transformer_learning_rate,
            weight_decay=self.config.transformer_weight_decay,
            warmup_ratio=self.config.transformer_warmup_ratio,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=max(10, steps_per_epoch // 10),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=cuda and not bf16,
            bf16=bf16,
            tf32=tf32,
            torch_compile=compile_enabled,
            optim="adamw_torch_fused" if cuda else "adamw_torch",
            dataloader_num_workers=0,
            dataloader_pin_memory=cuda,
            report_to=[],
            seed=self.config.seed,
            data_seed=self.config.seed,
        )
        weights = self.data.class_weights()
        trainer = FocalLossTrainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if cuda else None),
            compute_metrics=self._trainer_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.transformer_patience)],
            focal_gamma=self.config.focal_gamma,
            class_weights=(weights[0], weights[1]),
            label_smoothing=self.config.label_smoothing,
        )
        resume_checkpoint: str | None = None
        if self.config.resume_from_checkpoint == "auto":
            resume_checkpoint = get_last_checkpoint(str(run_dir)) if run_dir.exists() else None
            if resume_checkpoint:
                print(f"Checkpoint'ten devam ediliyor: {resume_checkpoint}")
            else:
                print("Devam edilecek checkpoint bulunamadi; egitim bastan basliyor.")
        elif self.config.resume_from_checkpoint is not None:
            checkpoint_path = Path(self.config.resume_from_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint bulunamadi: {checkpoint_path}")
            resume_checkpoint = str(checkpoint_path)
            print(f"Checkpoint'ten devam ediliyor: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)

        hf_dir = self.config.model_dir / "bert_best_model"
        trainer.save_model(str(hf_dir))
        tokenizer.save_pretrained(hf_dir)
        state_path = self.config.model_dir / "bert_best.pt"
        state_dict = {
            key.removeprefix("_orig_mod.").removeprefix("module."): value.detach().cpu()
            for key, value in trainer.model.state_dict().items()
        }
        torch.save(state_dict, state_path)

        validation_probs = self._probabilities(trainer.predict(valid_dataset))
        test_probs = self._probabilities(trainer.predict(test_dataset))
        bundle = self._calibrate_and_evaluate("bert", validation_probs, test_probs, hf_dir)
        self._write_manifest(bundle, hf_dir, state_path)
        del trainer, model
        if cuda:
            torch.cuda.empty_cache()
        gc.collect()
        return bundle

    def _write_manifest(self, bundle: PredictionBundle, hf_dir: Path, state_path: Path) -> None:
        data_version = self.data.dataset_manifest.get("data_version", "unknown")
        manifest = {
            "model_version": f"bert-tr-v3-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            "preprocessing_version": PREPROCESSING_VERSION,
            "threshold": round(float(np.clip(bundle.threshold, 0.5, 0.999)), 6),
            "temperature": round(bundle.temperature, 6),
            "label_mapping": {"0": "clean", "1": "toxic"},
            "training_data_version": data_version,
            "base_model": self.config.transformer_model,
            "max_length": self.config.transformer_max_length,
            "target_false_positive_rate": self.config.target_false_positive_rate,
            "validation_metrics": bundle.validation_metrics,
            "artifacts": {
                "bert_best.pt": file_sha256(state_path),
                "bert_best_model": directory_sha256(hf_dir),
            },
            "tokenizer_path": "bert_best_model",
        }
        manifest_path = self.config.model_dir / "model_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.config.deploy_backend:
            if PREPROCESSING_VERSION != BACKEND_PREPROCESSING_VERSION:
                raise RuntimeError(
                    "Belge (3) ön işlemesi backend tarafından henüz uygulanmıyor; "
                    "uyumsuz modeli otomatik dağıtmak güvenli değil."
                )
            backend_dir = self.config.project_dir / "backend_api"
            shutil.copy2(state_path, backend_dir / state_path.name)
            shutil.copy2(manifest_path, backend_dir / manifest_path.name)
            shutil.copytree(hf_dir, backend_dir / hf_dir.name, dirs_exist_ok=True)


class TrainingPipeline:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.data = TrainingDataModule(config)
        self.keras_resource_cache: dict[str, KerasTextResources] = {}

    def dry_run(self) -> dict[str, Any]:
        self.data.load()
        return {
            "status": "ready",
            "models": list(self.config.models),
            "rows": {name: len(frame) for name, frame in self.data.frames.items()},
            "class_weights": self.data.class_weights(),
            "data_version": self.data.dataset_manifest.get("data_version"),
        }

    def run(self) -> list[PredictionBundle]:
        self.config.result_dir.mkdir(parents=True, exist_ok=True)
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        # TensorFlow'u BERT'ten once import etmek Colab GPU bellegini rezerve
        # edebilir. Global seed asamasinda yalnizca kullanilacak Torch runtime'i
        # hazirlanir; Keras kendi egiticisinde, bellek buyumesinden sonra seedlenir.
        set_global_seed(
            self.config.seed,
            seed_torch="bert" in self.config.models,
            seed_tensorflow=False,
        )
        self.data.load()
        bundles: list[PredictionBundle] = []

        model_order = list(self.config.models)
        if "bert" in model_order and len(model_order) > 1:
            model_order.remove("bert")
            model_order.insert(0, "bert")
            print("Colab GPU bellegi icin BERT once, Keras modelleri sonra egitilecek.")

        for model_name in model_order:
            print(f"\n{'=' * 72}\n{model_name.upper()} eğitimi\n{'=' * 72}")
            if model_name == "bert":
                bundle = TransformerModelTrainer(self.config, self.data).train()
            else:
                bundle = KerasModelTrainer(
                    self.config,
                    self.data,
                    model_name,
                    self.keras_resource_cache,
                ).train()
            bundles.append(bundle)
            self._save_intermediate_results(bundles)

        self._save_intermediate_results(bundles)
        self._print_results(bundles)
        return bundles

    def _save_intermediate_results(self, bundles: Sequence[PredictionBundle]) -> None:
        rows: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        for bundle in bundles:
            row: dict[str, Any] = {"model": bundle.model_name}
            row.update({f"validation_{key}": value for key, value in bundle.validation_metrics.items()})
            row.update({f"test_{key}": value for key, value in bundle.test_metrics.items()})
            row["temperature"] = bundle.temperature
            row["artifact"] = bundle.artifact
            rows.append(row)
            details[bundle.model_name] = row
        pd.DataFrame(rows).to_csv(
            self.config.result_dir / "model_karsilastirma.csv", index=False, encoding="utf-8"
        )
        (self.config.result_dir / "model_metrics.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    @staticmethod
    def _print_results(bundles: Sequence[PredictionBundle]) -> None:
        print("\nModel karşılaştırması (model seçimi validation F1 ile)")
        for bundle in bundles:
            print(
                f"  {bundle.model_name:10s} | val F1={bundle.validation_metrics['f1']:.4f} "
                f"test F1={bundle.test_metrics['f1']:.4f} "
                f"test PR-AUC={bundle.test_metrics['pr_auc']:.4f} "
                f"eşik={bundle.threshold:.3f}"
            )
        best = max(bundles, key=lambda bundle: bundle.validation_metrics["f1"])
        print(f"\nValidation'a göre en iyi tek model: {best.model_name}")


def prepare_training_data(
    config: TrainingConfig,
    raw_data_dirs: Sequence[Path] = (),
    *,
    generate_plots: bool = False,
) -> None:
    """Ham CSV'lerden eğitim splitlerini aynı Python ortamında yeniden üretir.

    Bu seçenek özellikle Colab'da eski ``veri_setleri`` dosyalarının yanlışlıkla
    kullanılmasını engeller. Girdi klasörü verilmezse temizleme pipeline'ı proje
    kökünü, bir üst klasörü ve ``veri setleri ve url`` klasörünü tarar.
    """

    cleaner_script = SCRIPT_DIR / "01_veri_temizleme.py"
    if not cleaner_script.exists():
        raise FileNotFoundError(f"Veri temizleme betiği bulunamadı: {cleaner_script}")

    command = [
        sys.executable,
        str(cleaner_script),
        "--output-dir",
        str(config.data_dir),
        "--report-dir",
        str(config.result_dir / "veri_kalitesi"),
    ]
    for raw_data_dir in raw_data_dirs:
        command.extend(("--input-dir", str(Path(raw_data_dir).resolve())))
    if generate_plots:
        command.append("--plots")

    print(
        "Ham veriler temizleniyor ve train/valid/test splitleri yeniden oluşturuluyor...",
        flush=True,
    )
    subprocess.run(command, cwd=config.project_dir, check=True)

    required_outputs = [
        config.data_dir / "train.csv",
        config.data_dir / "valid.csv",
        config.data_dir / "test.csv",
        config.data_dir / "dataset_manifest.json",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"Veri hazırlama tamamlanmadı; eksik çıktılar: {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--models", default="lstm,cnn,cnn_lstm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--bert-model", default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--bert-epochs", type=float, default=3.0)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--keras-epochs", type=int, default=30)
    parser.add_argument("--keras-batch-size", type=int, default=64)
    parser.add_argument("--word2vec-epochs", type=int, default=10)
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Eğitimden/dry-run'dan önce tüm ham CSV'leri birleştirip splitleri yeniler.",
    )
    parser.add_argument(
        "--raw-data-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Ham CSV klasörü; birden çok kez verilebilir. Verilmezse proje kökü, "
            "bir üst klasör ve 'veri setleri ve url' otomatik taranır."
        ),
    )
    parser.add_argument(
        "--prepare-plots",
        action="store_true",
        help="--prepare-data sırasında veri analiz grafiklerini de üretir.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="auto",
        default=None,
        metavar="YOL",
        help="BERT egitimini verilen checkpoint'ten; yol verilmezse son checkpoint'ten surdur.",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--torch-compile",
        action="store_true",
        help="Desteklenen GPU'da torch.compile kullan (Colab T4 icin onerilmez).",
    )
    compile_group.add_argument(
        "--no-torch-compile",
        dest="torch_compile",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(torch_compile=False)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--deploy-backend", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Bağımlılık/model indirmeden split ve sızıntı kontrollerini çalıştırır.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        models=tuple(args.models.split(",")),
        seed=args.seed,
        target_false_positive_rate=args.target_fpr,
        transformer_model=args.bert_model,
        transformer_epochs=args.bert_epochs,
        transformer_batch_size=args.bert_batch_size,
        keras_epochs=args.keras_epochs,
        keras_batch_size=args.keras_batch_size,
        word2vec_epochs=args.word2vec_epochs,
        torch_compile=args.torch_compile,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
        deploy_backend=args.deploy_backend,
    )
    if args.prepare_data:
        prepare_training_data(
            config,
            raw_data_dirs=args.raw_data_dir,
            generate_plots=args.prepare_plots,
        )
    pipeline = TrainingPipeline(config)
    if args.dry_run:
        result = pipeline.dry_run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    return pipeline.run()


if __name__ == "__main__":
    main()
