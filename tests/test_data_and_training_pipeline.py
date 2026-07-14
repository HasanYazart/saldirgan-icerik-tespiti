from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")


ROOT = Path(__file__).resolve().parents[1]


def load_script(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def alphabetic_id(number: int) -> str:
    letters = []
    number += 1
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(ord("a") + remainder))
    return "".join(reversed(letters))


def test_cleaner_discovers_schemas_quarantines_conflicts_and_groups_near_duplicates(tmp_path):
    cleaning = load_script(
        "cleaning_pipeline_for_test",
        "veri_temizlemesi_ve_egitimi/01_veri_temizleme.py",
    )
    raw_dir = tmp_path / "ham_veri"
    output_dir = tmp_path / "veri_setleri"
    report_dir = tmp_path / "rapor"
    raw_dir.mkdir()

    first_rows = [
        {
            "message": (
                f"birinci {alphabetic_id(index)} kaynak sosyal medya "
                f"mesaji {alphabetic_id(index)}"
            ),
            "is_toxic": index % 2,
            "user_id": f"kullanici-{index // 2}",
        }
        for index in range(70)
    ]
    second_rows = [
        {
            "metin": (
                f"ikinci {alphabetic_id(index)} kaynak temizleme "
                f"ornegi {alphabetic_id(index)}"
            ),
            "etiket": "saldırgan" if index % 2 else "normal",
        }
        for index in range(70)
    ]
    first_rows.extend(
        [
            {"message": "bugün hava çok güzel arkadaşlar hep beraber", "is_toxic": 0},
            {"message": "aynı metin aynı etiket", "is_toxic": 1},
            {"message": "etiketi çelişkili metin", "is_toxic": 0},
        ]
    )
    second_rows.extend(
        [
            {"metin": "bugün hava güzel çok arkadaşlar hep beraber", "etiket": "normal"},
            {"metin": "aynı metin aynı etiket", "etiket": "saldırgan"},
            {"metin": "etiketi çelişkili metin", "etiket": "saldırgan"},
        ]
    )
    pd.DataFrame(first_rows).to_csv(raw_dir / "simulasyon.csv", index=False)
    pd.DataFrame(second_rows).to_csv(raw_dir / "eski_veri.csv", index=False)

    config = cleaning.DataCleaningConfig(
        project_dir=tmp_path,
        input_dirs=(raw_dir,),
        output_dir=output_dir,
        report_dir=report_dir,
        conflict_policy="quarantine",
        generate_plots=False,
    )
    audit = cleaning.TurkishToxicDataCleaner(config).run()

    assert audit["conflicting_texts"] == 1
    assert audit["entity_groups"] == 35
    assert audit["entity_grouped_rows"] == 35
    assert (report_dir / "etiket_celiskileri.csv").exists()
    assert (report_dir / "yinelenen_metinler.csv").exists()
    splits = {
        name: pd.read_csv(output_dir / f"{name}.csv")
        for name in ("train", "valid", "test")
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        assert set(splits[left]["text"]).isdisjoint(splits[right]["text"])
        assert set(splits[left]["group_id"]).isdisjoint(splits[right]["group_id"])

    combined = pd.concat(splits.values(), ignore_index=True)
    near = combined[combined["text"].str.contains("bugün hava")]
    assert len(near) == 2
    assert near["group_id"].nunique() == 1
    assert not combined["text"].str.contains("çelişkili").any()


def test_uploaded_multiclass_labels_and_base_key_groups_are_supported(tmp_path):
    cleaning = load_script(
        "cleaning_pipeline_uploaded_schema_test",
        "veri_temizlemesi_ve_egitimi/01_veri_temizleme.py",
    )
    raw_dir = tmp_path / "ham_veri"
    raw_dir.mkdir()
    rows = [
        {
            "text": "ornek temiz mesaj",
            "label": "clean",
            "base_key": "temel a",
            "agreement": "yes",
        },
        {
            "text": "hedefli saldiri ornegi",
            "label": "targeted_abuse",
            "base_key": "Temel B!",
            "agreement": "yes",
        },
        {
            "text": "cinsel kufur ornegi",
            "label": "sexual_profanity",
            "base_key": "Temel C!",
            "agreement": "yes",
        },
        {
            "text": "uzlasilmayan etiket",
            "label": "offensive",
            "base_key": "temel d",
            "agreement": "no",
        },
    ]
    pd.DataFrame(rows).to_csv(raw_dir / "train_2.csv", index=False)
    pd.DataFrame(rows[1:3]).to_csv(raw_dir / "test_2.csv", index=False)

    cleaner = cleaning.TurkishToxicDataCleaner(
        cleaning.DataCleaningConfig(project_dir=tmp_path, input_dirs=(raw_dir,))
    )
    train_frame = cleaner.load_and_standardize(raw_dir / "train_2.csv")
    test_frame = cleaner.load_and_standardize(raw_dir / "test_2.csv")

    assert train_frame["label"].tolist() == [0, 1, 1]
    assert cleaner.file_reports[0]["annotator_disagreement_rows"] == 1
    assert train_frame.loc[1, "entity_group"] == test_frame.loc[0, "entity_group"]
    assert cleaner._source_name(raw_dir / "train_2.csv") == "uploaded_2"
    assert cleaner._source_name(raw_dir / "test_2.csv") == "uploaded_2"


def test_manifested_training_data_is_not_normalized_twice(tmp_path):
    training = load_script(
        "training_pipeline_manifest_round_trip_test",
        "veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py",
    )
    data_dir = tmp_path / "veri_setleri"
    data_dir.mkdir()
    frames = {}
    for index, split in enumerate(("train", "valid", "test")):
        frame = pd.DataFrame(
            {
                "text": [
                    f'balonun icinde "#ia%" yaziyor {alphabetic_id(index)}',
                    f"normal mesaj {alphabetic_id(index)}",
                ],
                "label": [1, 0],
                "kaynak": ["uploaded_1", "uploaded_1"],
                "group_id": [f"g_{index}a", f"g_{index}b"],
            }
        )
        frames[split] = frame
        frame.to_csv(data_dir / f"{split}.csv", index=False)

    manifest = {
        "preprocessing_version": training.PREPROCESSING_VERSION,
        "split_fingerprints": {
            split: training.frame_fingerprint(frame) for split, frame in frames.items()
        },
    }
    (data_dir / "dataset_manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8"
    )

    module = training.TrainingDataModule(
        training.TrainingConfig(
            project_dir=tmp_path,
            data_dir=data_dir,
            models=("bert",),
        )
    ).load()
    assert module.train.loc[0, "text"] == 'balonun icinde "#ia%" yaziyor a'


def test_threshold_optimizer_respects_validation_false_positive_budget():
    training = load_script(
        "training_pipeline_for_test",
        "veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py",
    )
    labels = np.array([0] * 100 + [1] * 100)
    negative_probs = np.linspace(0.01, 0.72, 100)
    positive_probs = np.linspace(0.55, 0.99, 100)
    probabilities = np.concatenate([negative_probs, positive_probs])

    threshold, details = training.ThresholdOptimizer(target_fpr=0.05).select(
        probabilities, labels
    )
    metrics = training.calculate_metrics(labels, probabilities, threshold)

    assert 0.5 <= threshold <= 0.999
    assert details["false_positive_rate"] <= 0.05
    assert metrics["false_positive_rate"] <= 0.05
    assert metrics["f1"] > 0.5


def test_temperature_scaler_returns_finite_probabilities():
    training = load_script(
        "training_pipeline_temperature_test",
        "veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py",
    )
    probabilities = np.array([0.001, 0.05, 0.40, 0.60, 0.95, 0.999])
    labels = np.array([0, 0, 0, 1, 1, 1])
    scaler = training.TemperatureScaler().fit(probabilities, labels)
    calibrated = scaler.transform(probabilities)

    assert 0.349 <= scaler.temperature <= 4.001
    assert np.isfinite(calibrated).all()
    assert ((calibrated > 0) & (calibrated < 1)).all()
    assert np.all(np.diff(calibrated) > 0)


def test_training_parser_accepts_colab_data_preparation_flags():
    training = load_script(
        "training_pipeline_prepare_data_parser_test",
        "veri_temizlemesi_ve_egitimi/02_model_egitimi_colab.py",
    )
    args = training.build_parser().parse_args(
        [
            "--prepare-data",
            "--prepare-plots",
            "--raw-data-dir",
            "ham_veri_bir",
            "--raw-data-dir",
            "ham_veri_iki",
            "--models",
            "bert",
            "--dry-run",
        ]
    )

    assert args.prepare_data is True
    assert args.prepare_plots is True
    assert args.raw_data_dir == [Path("ham_veri_bir"), Path("ham_veri_iki")]
