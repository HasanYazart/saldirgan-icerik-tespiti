# -*- coding: utf-8 -*-
"""Türkçe saldırgan içerik verisi için sızıntı güvenli temizleme pipeline'ı.

Yeni ve eski CSV dosyalarını birlikte keşfeder, farklı sütun şemalarını tek bir
şemaya dönüştürür ve içerik gruplarını bölmeden train/valid/test üretir.

Örnek:
    python veri_temizlemesi_ve_egitimi/01_veri_temizleme.py --plots
    python veri_temizlemesi_ve_egitimi/01_veri_temizleme.py \
        --input-dir ham_veri --input-dir C:/veriler/simulasyon
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from veri_temizlemesi_ve_egitimi.document_preprocessing import (
    PREPROCESSING_VERSION,
    normalize_for_document,
)


TEXT_ALIASES = (
    "text",
    "metin",
    "message",
    "mesaj",
    "content",
    "icerik",
    "tweet",
    "comment",
    "yorum",
    "sentence",
    "cumle",
)
LABEL_ALIASES = (
    "is_toxic",
    "label",
    "toxic",
    "offensive",
    "is_offensive",
    "class",
    "sinif",
    "etiket",
    "target",
)
GROUP_ALIASES = (
    "base_key",
    "original_text",
    "root_text",
    "parent_text",
    "augmentation_group",
    "conversation_id",
    "thread_id",
    "user_id",
    "author_id",
    "account_id",
    "kullanici_id",
    "grup_id",
)
POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "evet",
    "toxic",
    "offensive",
    "abusive",
    "attack",
    "hate",
    "hakaret",
    "saldirgan",
    "saldırgan",
    "nefret",
    "insult",
    "threat",
    "profanity",
    "targeted abuse",
    "sexual profanity",
    "targeted harassment",
    "harassment",
    "cyberbullying",
    "off",
    "label 1",
}
NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "hayir",
    "hayır",
    "clean",
    "normal",
    "neutral",
    "not toxic",
    "non toxic",
    "not offensive",
    "non offensive",
    "not",
    "label 0",
}
SKIPPED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "modeller",
    "sonuclar",
    "grafikler",
    "veri_setleri",
    "temiz_veri",
    ".runtime",
    "reports",
    "rapor",
}

TEXTUAL_GROUP_ALIASES = {
    "base_key",
    "original_text",
    "root_text",
    "parent_text",
}
AGREEMENT_ALIASES = ("agreement", "annotator_agreement", "uzlasma", "uyum")
NOTES_ALIASES = ("notes", "note", "notlar", "aciklama")


def _ascii_identifier(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value).strip().lower())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    columns = [column for column in ("text", "label", "kaynak", "group_id") if column in frame]
    payload = frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class DataCleaningConfig:
    """Veri temizleme ve split davranışını tek noktadan yönetir."""

    project_dir: Path = PROJECT_DIR
    input_dirs: tuple[Path, ...] = field(default_factory=tuple)
    output_dir: Path | None = None
    report_dir: Path | None = None
    min_words: int = 1
    max_characters: int = 1500
    test_ratio: float = 0.15
    valid_ratio: float = 0.10
    random_seed: int = 42
    conflict_policy: str = "quarantine"
    group_near_duplicates: bool = True
    near_duplicate_ratio: float = 0.90
    generate_plots: bool = False

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir).resolve()
        self.output_dir = Path(self.output_dir or self.project_dir / "veri_setleri").resolve()
        self.report_dir = Path(self.report_dir or self.project_dir / "sonuclar" / "veri_kalitesi").resolve()
        self.input_dirs = tuple(Path(path).resolve() for path in self.input_dirs)
        if self.min_words < 1:
            raise ValueError("min_words en az 1 olmalı")
        if self.max_characters < 32:
            raise ValueError("max_characters en az 32 olmalı")
        if self.test_ratio <= 0 or self.valid_ratio <= 0:
            raise ValueError("test_ratio ve valid_ratio pozitif olmalı")
        if self.test_ratio + self.valid_ratio >= 0.5:
            raise ValueError("Validation + test oranı 0.50'den küçük olmalı")
        if self.conflict_policy not in {"quarantine", "error", "majority"}:
            raise ValueError("conflict_policy: quarantine, error veya majority olmalı")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class ContentGroupBuilder:
    """Biçim, kelime sırası ve tek kelimelik türevleri aynı grupta tutar.

    Amaç benzer satırları silmek değil, olası simülasyon/augmentation türevlerinin
    farklı split'lere düşmesini engellemektir. Böylece test skoru ezberlenmiş bir
    metnin küçük bir varyasyonuyla yapay olarak yükselmez.
    """

    def __init__(self, similarity_threshold: float = 0.90) -> None:
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _compact(text: str) -> str:
        return re.sub(r"[^\wçğıöşü]+", "", text, flags=re.IGNORECASE)

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[\wçğıöşü]+", text.lower(), flags=re.IGNORECASE))

    def _similar(self, left: str, right: str) -> bool:
        if left == right:
            return True
        left_tokens, right_tokens = set(self._tokens(left)), set(self._tokens(right))
        union = left_tokens | right_tokens
        jaccard = len(left_tokens & right_tokens) / max(len(union), 1)
        if jaccard >= self.similarity_threshold:
            return True
        return SequenceMatcher(None, left, right, autojunk=False).ratio() >= self.similarity_threshold

    @staticmethod
    def _remember_or_union(
        key: str,
        index: int,
        table: dict[str, int],
        union_find: UnionFind,
        texts: Sequence[str],
        comparator=None,
    ) -> None:
        previous = table.get(key)
        if previous is None:
            table[key] = index
        elif comparator is None or comparator(texts[previous], texts[index]):
            union_find.union(previous, index)

    def build(
        self, texts: Sequence[str], entity_groups: Sequence[str] | None = None
    ) -> tuple[list[str], dict[str, int]]:
        union_find = UnionFind(len(texts))
        compact_table: dict[str, int] = {}
        bag_table: dict[str, int] = {}
        deletion_table: dict[str, int] = {}
        token_sequence_table: dict[str, int] = {}
        entity_table: dict[str, int] = {}

        for index, text in enumerate(texts):
            compact = self._compact(text)
            if len(compact) >= 4:
                self._remember_or_union(compact, index, compact_table, union_find, texts)

            tokens = self._tokens(text)
            if 4 <= len(tokens) <= 14:
                full_token_key = " ".join(tokens)
                self._remember_or_union(
                    full_token_key,
                    index,
                    deletion_table,
                    union_find,
                    texts,
                    self._similar,
                )
                self._remember_or_union(
                    full_token_key,
                    index,
                    token_sequence_table,
                    union_find,
                    texts,
                    self._similar,
                )
            if 4 <= len(tokens) <= 40:
                bag_key = " ".join(sorted(tokens))
                self._remember_or_union(
                    bag_key, index, bag_table, union_find, texts, self._similar
                )

            # Özellikle sentetik random-word-deletion varyasyonlarını yakalar.
            if 5 <= len(tokens) <= 14:
                for removed_index in range(len(tokens)):
                    deletion_key = " ".join(tokens[:removed_index] + tokens[removed_index + 1 :])
                    self._remember_or_union(
                        deletion_key,
                        index,
                        token_sequence_table,
                        union_find,
                        texts,
                        self._similar,
                    )
                    self._remember_or_union(
                        deletion_key,
                        index,
                        deletion_table,
                        union_find,
                        texts,
                        self._similar,
                    )

        content_roots = [union_find.find(index) for index in range(len(texts))]
        near_duplicate_rows = len(content_roots) - len(set(content_roots))
        entity_grouped_rows = 0
        if entity_groups is not None:
            entity_counts: dict[str, int] = {}
            for index, entity_group in enumerate(entity_groups):
                if entity_group:
                    entity_counts[entity_group] = entity_counts.get(entity_group, 0) + 1
                    self._remember_or_union(
                        entity_group, index, entity_table, union_find, texts
                    )
            entity_grouped_rows = sum(count - 1 for count in entity_counts.values() if count > 1)

        roots = [union_find.find(index) for index in range(len(texts))]
        root_to_id: dict[int, str] = {}
        for root in sorted(set(roots)):
            # Harf icermeyen hash'lerin CSV okunurken sayiya donusup bastaki
            # sifirlarini kaybetmesini engellemek icin acik bir metin oneki.
            root_to_id[root] = "g_" + hashlib.blake2b(
                texts[root].encode("utf-8"), digest_size=8
            ).hexdigest()
        group_ids = [root_to_id[root] for root in roots]
        stats = {
            "groups": len(set(group_ids)),
            "near_duplicate_rows": int(near_duplicate_rows),
            "entity_groups": len(entity_table),
            "entity_grouped_rows": int(entity_grouped_rows),
        }
        return group_ids, stats


class TurkishToxicDataCleaner:
    """Tüm veri keşfi, temizliği, gruplaması ve split üretimini yürütür."""

    def __init__(self, config: DataCleaningConfig) -> None:
        self.config = config
        self.file_reports: list[dict[str, object]] = []
        self.rejected_frames: list[pd.DataFrame] = []
        self.conflicts = pd.DataFrame()
        self.duplicates = pd.DataFrame()

    def _default_input_dirs(self) -> tuple[Path, ...]:
        candidates = [
            self.config.project_dir / "ham_veri",
            self.config.project_dir / "raw_data",
            self.config.project_dir,
            self.config.project_dir.parent / "veri setleri ve url",
            self.config.project_dir.parent,
        ]
        env_dir = os.getenv("RAW_DATA_DIR")
        if env_dir:
            candidates.insert(0, Path(env_dir))
        resolved: list[Path] = []
        for path in candidates:
            path = path.resolve()
            if path.exists() and path not in resolved:
                resolved.append(path)
        return tuple(resolved)

    def discover_files(self) -> list[Path]:
        using_default_dirs = not self.config.input_dirs
        input_dirs = self.config.input_dirs or self._default_input_dirs()
        if not input_dirs:
            raise FileNotFoundError(
                "Ham veri klasörü bulunamadı. --input-dir verin veya ham_veri/ oluşturun."
            )

        output_dir = self.config.output_dir.resolve()
        files: set[Path] = set()
        shallow_default_dirs = {
            self.config.project_dir.resolve(),
            self.config.project_dir.parent.resolve(),
        }
        for input_dir in input_dirs:
            if not input_dir.exists():
                raise FileNotFoundError(f"Veri klasörü bulunamadı: {input_dir}")
            # Proje ve calisma alani koklerinde yalnizca dogrudan yuklenen
            # CSV'leri tara. ham_veri/raw_data gibi veri klasorleri ise
            # alt klasorleriyle birlikte kesfedilir. Boylece uretilmis split'ler
            # tekrar ham veri olarak okunmaz.
            iterator = (
                input_dir.glob("*.csv")
                if using_default_dirs and input_dir in shallow_default_dirs
                else input_dir.rglob("*.csv")
            )
            for path in iterator:
                resolved = path.resolve()
                if output_dir == resolved.parent or output_dir in resolved.parents:
                    continue
                if any(part.lower() in SKIPPED_DIRECTORIES for part in resolved.parts):
                    continue
                files.add(resolved)
        if not files:
            raise FileNotFoundError(f"CSV bulunamadı: {', '.join(map(str, input_dirs))}")
        return sorted(files, key=lambda path: str(path).casefold())

    @staticmethod
    def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []
        for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
            try:
                frame = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=None,
                    engine="python",
                    dtype=object,
                    on_bad_lines="warn",
                )
                if frame.shape[1] == 1:
                    raise ValueError("ayraç algılanamadı")
                return frame, encoding
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                errors.append(f"{encoding}: {exc}")
        raise ValueError(f"{path} okunamadı ({'; '.join(errors)})")

    @staticmethod
    def _find_column(columns: Iterable[object], aliases: Sequence[str]) -> object | None:
        normalized = {_ascii_identifier(column): column for column in columns}
        for alias in aliases:
            if alias in normalized:
                return normalized[alias]
        return None

    @staticmethod
    def _parse_label(value: object) -> int | None:
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            return int(numeric) if numeric in (0.0, 1.0) else None
        normalized = re.sub(r"[_-]+", " ", str(value).strip().lower())
        try:
            numeric = float(normalized.replace(",", "."))
            return int(numeric) if numeric in (0.0, 1.0) else None
        except ValueError:
            pass
        if normalized in POSITIVE_LABELS:
            return 1
        if normalized in NEGATIVE_LABELS:
            return 0
        return None

    @staticmethod
    def _source_name(path: Path) -> str:
        stem = _ascii_identifier(path.stem)
        split_number = re.fullmatch(r"(?:train|valid|validation|test)_(\d+)", stem)
        if split_number:
            return f"uploaded_{split_number.group(1)}"
        if "turkish_toxic" in stem:
            return "huggingface"
        if stem in {"train", "valid", "validation", "test"}:
            parent = _ascii_identifier(path.parent.name)
            return "kaggle" if parent in {"ham_veri", "veri_setleri_ve_url"} else parent
        return stem or "bilinmeyen"

    def load_and_standardize(self, path: Path) -> pd.DataFrame:
        raw, encoding = self._read_csv(path)
        text_column = self._find_column(raw.columns, TEXT_ALIASES)
        label_column = self._find_column(raw.columns, LABEL_ALIASES)
        group_column = self._find_column(raw.columns, GROUP_ALIASES)
        agreement_column = self._find_column(raw.columns, AGREEMENT_ALIASES)
        notes_column = self._find_column(raw.columns, NOTES_ALIASES)
        if text_column is None or label_column is None:
            raise ValueError(
                f"{path}: metin/etiket sütunu bulunamadı. Sütunlar={list(raw.columns)}"
            )

        source = self._source_name(path)
        standardized = pd.DataFrame(
            {
                "raw_text": raw[text_column],
                "raw_label": raw[label_column],
                "kaynak": source,
                "dosya": path.name,
                "satir_no": np.arange(2, len(raw) + 2),
            }
        )
        group_column_id = _ascii_identifier(group_column) if group_column is not None else ""
        if group_column is None:
            standardized["entity_group"] = ""
        else:
            def stable_group_value(value: object) -> str:
                if pd.isna(value) or str(value).strip() == "":
                    return ""
                raw_value = str(value).strip()
                if group_column_id in TEXTUAL_GROUP_ALIASES:
                    normalized_value = normalize_for_document(raw_value)
                else:
                    normalized_value = raw_value.casefold()
                return source + ":" + hashlib.blake2b(
                    normalized_value.encode("utf-8"), digest_size=8
                ).hexdigest()

            standardized["entity_group"] = raw[group_column].map(
                stable_group_value
            )
        standardized["label"] = standardized["raw_label"].map(self._parse_label)
        standardized["text"] = standardized["raw_text"].map(normalize_for_document)
        standardized["kelime_sayisi"] = standardized["text"].str.split().str.len()

        reasons = np.full(len(standardized), "", dtype=object)
        reasons[standardized["raw_text"].isna().to_numpy()] = "null_text"
        reasons[(standardized["text"] == "").to_numpy()] = "empty_after_normalization"
        reasons[standardized["label"].isna().to_numpy()] = "invalid_label"
        disagreement_mask = np.zeros(len(standardized), dtype=bool)
        if agreement_column is not None:
            agreement = raw[agreement_column].map(
                lambda value: re.sub(r"[_-]+", " ", str(value).strip().lower())
                if not pd.isna(value)
                else ""
            )
            disagreement_mask = agreement.isin(
                {"0", "false", "no", "hayir", "hayır", "disagree", "uyusmuyor"}
            ).to_numpy()
            reasons[(reasons == "") & disagreement_mask] = "annotator_disagreement"
        too_short = standardized["kelime_sayisi"].fillna(0).lt(self.config.min_words).to_numpy()
        reasons[(reasons == "") & too_short] = "too_short"
        valid_mask = reasons == ""

        rejected = standardized.loc[~valid_mask].copy()
        if not rejected.empty:
            rejected["red_nedeni"] = reasons[~valid_mask]
            self.rejected_frames.append(rejected)

        clean = standardized.loc[valid_mask].copy()
        long_mask = clean["text"].str.len().gt(self.config.max_characters)
        clean.loc[long_mask, "text"] = clean.loc[long_mask, "text"].str.slice(
            stop=self.config.max_characters
        ).str.rsplit(" ", n=1).str[0]
        clean["label"] = clean["label"].astype("int8")

        self.file_reports.append(
            {
                "path": str(path),
                "source": source,
                "encoding": encoding,
                "text_column": str(text_column),
                "label_column": str(label_column),
                "group_column": str(group_column) if group_column is not None else None,
                "agreement_column": (
                    str(agreement_column) if agreement_column is not None else None
                ),
                "raw_rows": int(len(raw)),
                "accepted_rows": int(len(clean)),
                "rejected_rows": int(len(rejected)),
                "annotator_disagreement_rows": int(disagreement_mask.sum()),
                "placeholder_annotation_rows": int(
                    raw[notes_column]
                    .fillna("")
                    .astype(str)
                    .str.contains("placeholder", case=False, regex=False)
                    .sum()
                    if notes_column is not None
                    else 0
                ),
                "raw_label_distribution": {
                    str(key): int(value)
                    for key, value in raw[label_column]
                    .fillna("<NULL>")
                    .astype(str)
                    .value_counts()
                    .items()
                },
                "binary_label_distribution": {
                    str(key): int(value)
                    for key, value in clean["label"].value_counts().sort_index().items()
                },
                "truncated_rows": int(long_mask.sum()),
                "sha256": _sha256_file(path),
            }
        )
        return clean[["text", "label", "kaynak", "dosya", "satir_no", "entity_group"]]

    def resolve_duplicates_and_conflicts(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(["text", "label", "kaynak", "dosya", "satir_no"]).reset_index(drop=True)
        conflict_texts = frame.groupby("text", sort=False)["label"].nunique()
        conflict_texts = set(conflict_texts[conflict_texts > 1].index)
        self.conflicts = frame[frame["text"].isin(conflict_texts)].copy()

        if conflict_texts and self.config.conflict_policy == "error":
            self._write_quality_tables()
            raise ValueError(
                f"{len(conflict_texts)} çelişkili metin bulundu: "
                f"{self.config.report_dir / 'etiket_celiskileri.csv'}"
            )
        if conflict_texts and self.config.conflict_policy == "quarantine":
            frame = frame[~frame["text"].isin(conflict_texts)].copy()
        elif conflict_texts and self.config.conflict_policy == "majority":
            counts = frame.groupby(["text", "label"]).size().unstack(fill_value=0)
            unambiguous = counts[counts.max(axis=1) > counts.sum(axis=1) / 2].idxmax(axis=1)
            selected = frame[frame["text"].isin(unambiguous.index)].copy()
            selected = selected[selected.apply(lambda row: row["label"] == unambiguous[row["text"]], axis=1)]
            frame = pd.concat(
                [frame[~frame["text"].isin(conflict_texts)], selected], ignore_index=True
            )

        duplicate_mask = frame.duplicated("text", keep=False)
        self.duplicates = frame[duplicate_mask].copy()
        return frame.drop_duplicates("text", keep="first").reset_index(drop=True)

    def add_content_groups(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        frame = frame.sort_values(["text", "label", "kaynak"]).reset_index(drop=True)
        if self.config.group_near_duplicates:
            group_ids, stats = ContentGroupBuilder(
                self.config.near_duplicate_ratio
            ).build(frame["text"].tolist(), frame["entity_group"].tolist())
        else:
            group_ids = [
                "g_" + hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
                for text in frame["text"]
            ]
            stats = {
                "groups": len(group_ids),
                "near_duplicate_rows": 0,
                "entity_groups": 0,
                "entity_grouped_rows": 0,
            }
        frame["group_id"] = group_ids
        return frame, stats

    def _fold_count(self) -> int:
        denominators = [
            Fraction(self.config.test_ratio).limit_denominator(100).denominator,
            Fraction(self.config.valid_ratio).limit_denominator(100).denominator,
        ]
        return min(math.lcm(*denominators), 50)

    def split(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        fold_count = self._fold_count()
        group_count = frame["group_id"].nunique()
        if group_count < fold_count:
            raise ValueError(f"Split için en az {fold_count} içerik grubu gerekli; bulunan={group_count}")

        source_strata = frame["label"].astype(str) + "|" + frame["kaynak"].astype(str)
        rare = source_strata.map(source_strata.value_counts()).lt(fold_count)
        strata = source_strata.mask(rare, frame["label"].astype(str) + "|diger_kaynak")
        if strata.map(strata.value_counts()).min() < fold_count:
            strata = frame["label"].astype(str)

        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        fold_ids = np.empty(len(frame), dtype=np.int16)
        for fold_id, (_, validation_indices) in enumerate(
            splitter.split(frame["text"], strata, groups=frame["group_id"])
        ):
            fold_ids[validation_indices] = fold_id

        test_fold_count = max(1, round(self.config.test_ratio * fold_count))
        valid_fold_count = max(1, round(self.config.valid_ratio * fold_count))
        test_folds = set(range(test_fold_count))
        valid_folds = set(range(test_fold_count, test_fold_count + valid_fold_count))
        masks = {
            "test": np.isin(fold_ids, list(test_folds)),
            "valid": np.isin(fold_ids, list(valid_folds)),
        }
        masks["train"] = ~(masks["test"] | masks["valid"])
        splits = {
            name: frame.loc[mask, ["text", "label", "kaynak", "group_id"]]
            .sample(frac=1, random_state=self.config.random_seed)
            .reset_index(drop=True)
            for name, mask in masks.items()
        }
        self._assert_no_leakage(splits)
        return splits

    @staticmethod
    def _assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            text_overlap = set(splits[left]["text"]) & set(splits[right]["text"])
            group_overlap = set(splits[left]["group_id"]) & set(splits[right]["group_id"])
            if text_overlap or group_overlap:
                raise RuntimeError(
                    f"Veri sızıntısı: {left}-{right}; text={len(text_overlap)}, group={len(group_overlap)}"
                )

    def _write_quality_tables(self) -> None:
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        if not self.conflicts.empty:
            self.conflicts.to_csv(
                self.config.report_dir / "etiket_celiskileri.csv", index=False, encoding="utf-8"
            )
        if not self.duplicates.empty:
            self.duplicates.to_csv(
                self.config.report_dir / "yinelenen_metinler.csv", index=False, encoding="utf-8"
            )
        if self.rejected_frames:
            pd.concat(self.rejected_frames, ignore_index=True).to_csv(
                self.config.report_dir / "reddedilen_satirlar.csv", index=False, encoding="utf-8"
            )

    def _generate_plots(self, combined: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_dir = self.config.project_dir / "grafikler"
        plot_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams["font.family"] = "DejaVu Sans"

        def save(figure: object, filename: str) -> None:
            figure.tight_layout()
            figure.savefig(plot_dir / filename, dpi=170, bbox_inches="tight")
            plt.close(figure)

        plot_frame = combined[["text", "label", "kaynak", "group_id"]].copy()
        plot_frame["sinif"] = plot_frame["label"].map({0: "Temiz", 1: "Saldırgan"})
        plot_frame["karakter_sayisi"] = plot_frame["text"].str.len()
        plot_frame["kelime_sayisi"] = plot_frame["text"].str.split().str.len()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.countplot(data=plot_frame, x="sinif", ax=axes[0], hue="sinif", legend=False)
        axes[0].set_title("Sınıf dağılımı")
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt="%.0f")
        source_counts = plot_frame["kaynak"].value_counts().head(15)
        sns.barplot(x=source_counts.values, y=source_counts.index, ax=axes[1], color="#4c72b0")
        axes[1].set_title("Kaynak dağılımı")
        save(fig, "01_sinif_ve_kaynak_dagilimi.png")

        split_frame = pd.concat(
            [part.assign(split=name) for name, part in splits.items()], ignore_index=True
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.countplot(data=split_frame, x="split", hue="label", ax=ax)
        ax.set_title("Split ve sınıf dağılımı")
        ax.legend(title="Etiket", labels=["Temiz", "Saldırgan"])
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f")
        save(fig, "02_split_dagilimi.png")

        character_limit = max(float(plot_frame["karakter_sayisi"].quantile(0.99)), 1)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.histplot(
            data=plot_frame[plot_frame["karakter_sayisi"] <= character_limit],
            x="karakter_sayisi",
            hue="sinif",
            bins=60,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_title("Metin karakter uzunluğu dağılımı (%99 aralık)")
        save(fig, "03_metin_karakter_uzunlugu.png")

        word_limit = max(float(plot_frame["kelime_sayisi"].quantile(0.99)), 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(
            data=plot_frame[plot_frame["kelime_sayisi"] <= word_limit],
            x="kelime_sayisi",
            hue="sinif",
            bins=50,
            stat="density",
            common_norm=False,
            element="step",
            ax=axes[0],
        )
        axes[0].set_title("Kelime sayısı dağılımı (%99 aralık)")
        sns.boxplot(
            data=plot_frame[plot_frame["kelime_sayisi"] <= word_limit],
            x="sinif",
            y="kelime_sayisi",
            hue="sinif",
            legend=False,
            ax=axes[1],
        )
        axes[1].set_title("Sınıfa göre kelime sayısı")
        save(fig, "04_kelime_uzunlugu_histogram_boxplot.png")

        overall_words = Counter(
            token for text in plot_frame["text"] for token in str(text).split()
        ).most_common(30)
        fig, ax = plt.subplots(figsize=(10, 8))
        if overall_words:
            words, counts = zip(*reversed(overall_words))
            ax.barh(words, counts, color="#55a868")
        ax.set_title("En sık 30 kelime")
        ax.set_xlabel("Frekans")
        save(fig, "05_en_sik_kelimeler.png")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for label, axis, color in ((0, axes[0], "#4c72b0"), (1, axes[1], "#c44e52")):
            counts = Counter(
                token
                for text in plot_frame.loc[plot_frame["label"] == label, "text"]
                for token in str(text).split()
            ).most_common(25)
            if counts:
                words, values = zip(*reversed(counts))
                axis.barh(words, values, color=color)
            axis.set_title(f"{('Temiz' if label == 0 else 'Saldırgan')} sınıfında sık kelimeler")
        save(fig, "06_sinif_bazli_en_sik_kelimeler.png")

        try:
            from wordcloud import WordCloud

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            for label, axis, color_map in ((0, axes[0], "Blues"), (1, axes[1], "Reds")):
                texts = plot_frame.loc[plot_frame["label"] == label, "text"]
                if len(texts) > 50_000:
                    texts = texts.sample(50_000, random_state=self.config.random_seed)
                cloud = WordCloud(
                    width=1200,
                    height=700,
                    background_color="white",
                    colormap=color_map,
                    max_words=250,
                    collocations=False,
                    random_state=self.config.random_seed,
                ).generate(" ".join(texts.astype(str)))
                axis.imshow(cloud, interpolation="bilinear")
                axis.axis("off")
                axis.set_title("Temiz WordCloud" if label == 0 else "Saldırgan WordCloud")
            save(fig, "07_sinif_bazli_wordcloud.png")
        except (ImportError, ValueError) as exc:
            print(f"WordCloud üretilemedi: {exc}")

        source_label = pd.crosstab(
            plot_frame["kaynak"], plot_frame["sinif"], normalize="index"
        )
        fig, ax = plt.subplots(figsize=(10, max(4, 0.65 * len(source_label))))
        sns.heatmap(source_label, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax)
        ax.set_title("Kaynak bazında sınıf oranları")
        save(fig, "08_kaynak_sinif_oranlari.png")

        source_absolute = pd.crosstab(plot_frame["kaynak"], plot_frame["sinif"])
        fig, ax = plt.subplots(figsize=(11, max(4, 0.65 * len(source_absolute))))
        source_absolute.plot(kind="barh", stacked=True, ax=ax, color=["#4c72b0", "#c44e52"])
        ax.set_title("Kaynak bazında mutlak sınıf sayıları")
        ax.set_xlabel("Satır")
        save(fig, "09_kaynak_sinif_sayilari.png")

        group_sizes = plot_frame.groupby("group_id").size()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(group_sizes.clip(upper=group_sizes.quantile(0.99)), bins=40, ax=axes[0])
        axes[0].set_title("Yakın-kopya grup boyutu (%99 aralık)")
        axes[0].set_xlabel("Gruptaki metin sayısı")
        group_summary = pd.Series(
            {
                "Tekil grup": int((group_sizes == 1).sum()),
                "Çoklu grup": int((group_sizes > 1).sum()),
            }
        )
        axes[1].pie(group_summary.values, labels=group_summary.index, autopct="%1.1f%%")
        axes[1].set_title("İçerik grupları")
        save(fig, "10_yakin_kopya_gruplari.png")

        file_quality = pd.DataFrame(self.file_reports)
        if not file_quality.empty:
            file_quality["dosya"] = file_quality["path"].map(lambda value: Path(value).name)
            fig, ax = plt.subplots(figsize=(12, max(5, 0.7 * len(file_quality))))
            file_quality.set_index("dosya")[["accepted_rows", "rejected_rows"]].plot(
                kind="barh",
                stacked=True,
                color=["#55a868", "#c44e52"],
                ax=ax,
            )
            ax.set_title("Dosya bazında kabul/reddedilme")
            ax.set_xlabel("Satır")
            save(fig, "11_dosya_kalite_ozeti.png")

        rejected_count = sum(len(frame) for frame in self.rejected_frames)
        quality_counts = pd.Series(
            {
                "Eğitime kabul": len(combined),
                "Yinelenen kayıt": len(self.duplicates),
                "Çelişkili metin": self.conflicts["text"].nunique()
                if not self.conflicts.empty
                else 0,
                "Geçersiz/reddedilen": rejected_count,
            }
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(quality_counts.index, quality_counts.values, color=sns.color_palette("Set2", 4))
        ax.bar_label(bars, fmt="%.0f")
        ax.set_title("Veri temizleme kalite özeti")
        ax.tick_params(axis="x", rotation=15)
        save(fig, "12_temizleme_kalite_ozeti.png")

        print(f"{len(list(plot_dir.glob('*.png')))} veri grafiği üretildi: {plot_dir}")

    def _build_audit(
        self,
        combined: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        group_stats: dict[str, int],
    ) -> dict[str, object]:
        report: dict[str, object] = {
            "preprocessing_version": PREPROCESSING_VERSION,
            "seed": self.config.random_seed,
            "input_files": self.file_reports,
            "accepted_unique_rows": int(len(combined)),
            "exact_duplicate_rows": int(len(self.duplicates)),
            "conflicting_texts": int(self.conflicts["text"].nunique()) if not self.conflicts.empty else 0,
            **group_stats,
            "splits": {},
            "overlap": {},
        }
        for name, part in splits.items():
            report["splits"][name] = {
                "rows": int(len(part)),
                "groups": int(part["group_id"].nunique()),
                "labels": {str(key): int(value) for key, value in part["label"].value_counts().items()},
                "sources": {str(key): int(value) for key, value in part["kaynak"].value_counts().items()},
                "sha256": _frame_fingerprint(part),
            }
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            report["overlap"][f"{left}-{right}"] = {
                "text": len(set(splits[left]["text"]) & set(splits[right]["text"])),
                "group": len(set(splits[left]["group_id"]) & set(splits[right]["group_id"])),
            }
        version_payload = "".join(
            sorted(item["sha256"] for item in self.file_reports)
        ) + PREPROCESSING_VERSION
        report["data_version"] = "combined-" + hashlib.sha256(version_payload.encode()).hexdigest()[:12]
        return report

    def run(self) -> dict[str, object]:
        files = self.discover_files()
        print(f"\n{len(files)} ham CSV bulundu:")
        for path in files:
            print(f"  - {path}")

        frames = [self.load_and_standardize(path) for path in files]
        combined = self.resolve_duplicates_and_conflicts(pd.concat(frames, ignore_index=True))
        combined, group_stats = self.add_content_groups(combined)
        splits = self.split(combined)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            part.to_csv(self.config.output_dir / f"{name}.csv", index=False, encoding="utf-8")
        combined.to_csv(
            self.config.output_dir / "birlesik_tum_veri.csv", index=False, encoding="utf-8"
        )
        self._write_quality_tables()

        audit = self._build_audit(combined, splits, group_stats)
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        (self.config.report_dir / "data_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.config.output_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "data_version": audit["data_version"],
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "split_fingerprints": {
                        name: audit["splits"][name]["sha256"] for name in splits
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.config.generate_plots:
            self._generate_plots(combined, splits)

        print("\nTemizleme tamamlandı:")
        for name in ("train", "valid", "test"):
            print(f"  {name:5s}: {len(splits[name]):,}")
        print(f"  veri sürümü: {audit['data_version']}")
        print(f"  kalite raporu: {self.config.report_dir / 'data_audit.json'}")
        return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        default=[],
        help="Birden çok kez verilebilir. Verilmezse ham_veri ve eski veri klasörü aranır.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--max-characters", type=int, default=1500)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conflict-policy", choices=("quarantine", "error", "majority"), default="quarantine"
    )
    parser.add_argument("--no-near-duplicate-groups", action="store_true")
    parser.add_argument("--near-duplicate-ratio", type=float, default=0.90)
    parser.add_argument("--plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    config = DataCleaningConfig(
        input_dirs=tuple(args.input_dir),
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        min_words=args.min_words,
        max_characters=args.max_characters,
        test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio,
        random_seed=args.seed,
        conflict_policy=args.conflict_policy,
        group_near_duplicates=not args.no_near_duplicate_groups,
        near_duplicate_ratio=args.near_duplicate_ratio,
        generate_plots=args.plots,
    )
    return TurkishToxicDataCleaner(config).run()


if __name__ == "__main__":
    main()
