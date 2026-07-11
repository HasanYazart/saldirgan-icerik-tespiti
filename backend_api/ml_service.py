"""Kural katmanı ve sürümlenmiş BERT modelini birleştiren servis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .nlp_service import full_analysis
from .settings import settings
from .text_processing import PREPROCESSING_VERSION

try:
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    ML_DEPENDENCIES_AVAILABLE = True
except ImportError:
    torch: Any = None
    BertForSequenceClassification: Any = None
    BertTokenizer: Any = None
    ML_DEPENDENCIES_AVAILABLE = False


MODEL_NAME = "dbmdz/bert-base-turkish-cased"
DEFAULT_TOXICITY_THRESHOLD = 0.80


class MLService:
    def __init__(self) -> None:
        self.device: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch else "cpu"
        self.tokenizer: Any = None
        self.model: Any = None
        self.is_mock = True
        self.degraded_reason: str | None = None
        self.model_version = "rules-only"
        self.toxicity_threshold = DEFAULT_TOXICITY_THRESHOLD
        self.manifest: dict = {}
        self._load_model()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _directory_sha256(cls, directory: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
            digest.update(bytes.fromhex(cls._sha256(path)))
        return digest.hexdigest()

    def _load_manifest(self) -> bool:
        path = settings.model_dir / "model_manifest.json"
        if not path.exists():
            self.degraded_reason = "model_manifest.json bulunamadı; doğrulanmamış model yüklenmedi"
            return False
        try:
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
            required = {"model_version", "preprocessing_version", "threshold", "label_mapping", "artifacts"}
            missing = required - self.manifest.keys()
            if missing:
                raise ValueError(f"eksik alanlar: {', '.join(sorted(missing))}")
            if self.manifest["preprocessing_version"] != PREPROCESSING_VERSION:
                raise ValueError("eğitim ve servis ön işleme sürümleri farklı")
            threshold = float(self.manifest["threshold"])
            if not 0.5 <= threshold <= 0.999:
                raise ValueError("threshold 0.5-0.999 aralığında olmalı")
            if self.manifest["label_mapping"] != {"0": "clean", "1": "toxic"}:
                raise ValueError("desteklenmeyen label_mapping")
            self.toxicity_threshold = threshold
            self.model_version = str(self.manifest["model_version"])
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            self.degraded_reason = f"geçersiz model manifesti: {error}"
            return False

    def _load_model(self) -> None:
        if not ML_DEPENDENCIES_AVAILABLE:
            self.degraded_reason = "torch/transformers kurulu değil"
            return
        if not self._load_manifest():
            return

        hf_model_path = settings.model_dir / "bert_best_model"
        pt_model_path = settings.model_dir / "bert_best.pt"
        try:
            if hf_model_path.is_dir():
                expected_hash = self.manifest["artifacts"].get("bert_best_model")
                if not expected_hash or self._directory_sha256(hf_model_path) != expected_hash:
                    raise ValueError("HuggingFace model klasörü SHA-256 doğrulaması başarısız")
                self.tokenizer = BertTokenizer.from_pretrained(hf_model_path, local_files_only=True)
                self.model = BertForSequenceClassification.from_pretrained(hf_model_path, local_files_only=True)
            elif pt_model_path.exists():
                expected_hash = self.manifest["artifacts"].get("bert_best.pt")
                if not expected_hash or self._sha256(pt_model_path) != expected_hash:
                    raise ValueError("model SHA-256 doğrulaması başarısız")
                tokenizer_source = self.manifest.get("tokenizer_path")
                if not tokenizer_source:
                    raise ValueError(".pt kullanımı için manifestte tokenizer_path zorunlu")
                tokenizer_path = (settings.model_dir / tokenizer_source).resolve()
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                self.model = BertForSequenceClassification.from_pretrained(
                    tokenizer_path, num_labels=2, local_files_only=True
                )
                state_dict = torch.load(pt_model_path, map_location=self.device, weights_only=True)
                state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
                self.model.load_state_dict(state_dict, strict=True)
            else:
                raise FileNotFoundError("bert_best_model/ veya bert_best.pt bulunamadı")

            self.model.to(self.device)
            self.model.eval()
            self.is_mock = False
            self.degraded_reason = None
        except Exception as error:
            self.model = None
            self.tokenizer = None
            self.is_mock = True
            self.degraded_reason = str(error)

    def ensure_available(self) -> None:
        if self.is_mock and not settings.allow_mock_model:
            raise RuntimeError(f"Model kullanılamıyor: {self.degraded_reason}")

    def _bert_predict(self, text: str) -> float:
        with torch.inference_mode():
            encoding = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            )
            return torch.softmax(outputs.logits.float(), dim=1)[0, 1].item()

    @staticmethod
    def _category(text: str, method: str) -> tuple[str, str]:
        lowered = text.lower()
        if any(term in lowered for term in ("öldür", "gebert", "bıçak", "silah", "kafanı kır")):
            return "threat", "critical"
        if method in {"dictionary_evasion", "message_buffer"}:
            return "targeted_abuse", "high"
        return "offensive_language", "medium"

    def analyze_text(self, text: str, user_id: int = 0) -> dict:
        nlp_result = full_analysis(text, user_id=user_id)
        if nlp_result["dictionary_hit"] or nlp_result["buffer_hit"]:
            method = "dictionary_evasion" if nlp_result["dictionary_hit"] else "message_buffer"
            category, severity = self._category(text, method)
            return {
                "is_toxic": True,
                "needs_review": False,
                "toxicity_score": 0.99 if method == "dictionary_evasion" else 0.95,
                "detection_method": method,
                "preprocessed_text": nlp_result["preprocessed_text"],
                "category": category,
                "severity": severity,
                "model_version": self.model_version,
            }

        if self.is_mock:
            return {
                "is_toxic": False,
                "needs_review": False,
                "toxicity_score": 0.0,
                "detection_method": "rules_only_clean",
                "preprocessed_text": nlp_result["preprocessed_text"],
                "category": "clean",
                "severity": "none",
                "model_version": "rules-only",
            }

        score = self._bert_predict(nlp_result["preprocessed_text"])
        is_toxic = score >= self.toxicity_threshold
        review_floor = max(0.50, self.toxicity_threshold - 0.10)
        needs_review = not is_toxic and score >= review_floor
        category, severity = self._category(text, "bert_calibrated") if (is_toxic or needs_review) else ("clean", "none")
        return {
            "is_toxic": is_toxic,
            "needs_review": needs_review,
            "toxicity_score": round(score, 4),
            "detection_method": "bert_calibrated",
            "preprocessed_text": nlp_result["preprocessed_text"],
            "category": category,
            "severity": severity,
            "model_version": self.model_version,
        }

    def health(self) -> dict:
        return {
            "ready": not self.is_mock,
            "mode": "bert" if not self.is_mock else "rules_only",
            "reason": self.degraded_reason,
            "model_version": self.model_version,
            "preprocessing_version": PREPROCESSING_VERSION,
            "threshold": self.toxicity_threshold,
            "device": str(self.device),
        }


ml_service = MLService()
