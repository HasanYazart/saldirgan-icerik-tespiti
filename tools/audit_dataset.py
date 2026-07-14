"""Train/validation/test veri kalitesini ölçer ve JSON raporu üretir."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def fingerprint_frame(frame: pd.DataFrame) -> str:
    rows = frame[["text", "label"]].sort_values(["text", "label"]).to_csv(index=False)
    return hashlib.sha256(rows.encode("utf-8")).hexdigest()


def audit(data_dir: Path) -> dict:
    frames = {}
    report = {"splits": {}, "overlap": {}, "group_overlap": {}, "label_conflicts": 0}
    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}.csv"
        frame = pd.read_csv(path)
        missing = {"text", "label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: eksik sütunlar {sorted(missing)}")
        frame = frame[
            [column for column in ("text", "label", "kaynak", "group_id") if column in frame.columns]
        ]
        frames[split] = frame
        report["splits"][split] = {
            "rows": len(frame),
            "null_texts": int(frame["text"].isna().sum()),
            "duplicate_texts": int(frame.duplicated("text").sum()),
            "short_messages": int(frame["text"].fillna("").str.split().str.len().le(2).sum()),
            "labels": {str(key): int(value) for key, value in frame["label"].value_counts().items()},
            "sha256": fingerprint_frame(frame),
        }
        if "kaynak" in frame:
            report["splits"][split]["sources"] = {
                str(key): int(value) for key, value in frame["kaynak"].value_counts().items()
            }
        if "group_id" in frame:
            report["splits"][split]["groups"] = int(frame["group_id"].nunique())

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        report["overlap"][f"{left}-{right}"] = len(
            set(frames[left]["text"].dropna()) & set(frames[right]["text"].dropna())
        )
        if "group_id" in frames[left] and "group_id" in frames[right]:
            report["group_overlap"][f"{left}-{right}"] = len(
                set(frames[left]["group_id"].dropna())
                & set(frames[right]["group_id"].dropna())
            )

    combined = pd.concat(frames.values(), ignore_index=True)
    report["label_conflicts"] = int((combined.groupby("text")["label"].nunique() > 1).sum())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("veri_setleri"))
    parser.add_argument("--output", type=Path, default=Path("sonuclar/data_audit.json"))
    args = parser.parse_args()
    report = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (
        any(report["overlap"].values())
        or any(report["group_overlap"].values())
        or report["label_conflicts"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
