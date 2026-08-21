from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.benchmarks.data.multiban import MultiBanFakeDetectDataset
from src.benchmarks.preflight import prepare_binary_benchmark
from src.benchmarks.text_normalization import (
    BANGLA_NORMALIZER_PROVIDER,
    BenchmarkTextNormalizer,
    text_normalization_signature,
)
from src.utils.config import load_yaml_config


def _config() -> dict:
    return {
        "dataset": {
            "text": {
                "normalization": {
                    "enabled": True,
                    "provider": BANGLA_NORMALIZER_PROVIDER,
                    "unicode_norm": "NFKC",
                    "punct_replacement": None,
                    "url_replacement": None,
                    "emoji_replacement": None,
                    "apply_unicode_norm_last": True,
                }
            }
        }
    }


def test_banglabert_normalizer_forwards_the_reproducible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_normalize(text: str, **kwargs: object) -> str:
        captured["text"] = text
        captured.update(kwargs)
        return "normalized bangla"

    module = types.ModuleType("normalizer")
    module.normalize = fake_normalize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "normalizer", module)

    normalizer = BenchmarkTextNormalizer.from_config(
        _config()["dataset"]["text"]["normalization"]
    )

    assert normalizer("raw bangla") == "normalized bangla"
    assert captured == {
        "text": "raw bangla",
        "unicode_norm": "NFKC",
        "punct_replacement": None,
        "url_replacement": None,
        "emoji_replacement": None,
        "apply_unicode_norm_last": True,
    }
    assert text_normalization_signature(_config()) == normalizer.signature()


def test_disabled_text_normalizer_is_an_identity_without_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "normalizer", raising=False)
    normalizer = BenchmarkTextNormalizer.from_config(
        {"enabled": False, "provider": "none"}
    )

    assert normalizer(" unchanged ") == " unchanged "


def test_unknown_enabled_normalizer_provider_fails_closed() -> None:
    normalizer = BenchmarkTextNormalizer.from_config(
        {"enabled": True, "provider": "unreviewed/provider"}
    )

    with pytest.raises(ValueError, match="Unsupported benchmark text normalizer"):
        normalizer("text")


class _FakeEncoding(dict):
    def word_ids(self) -> list[int | None]:
        return [None, 0, 1, None]


class _FakeFastTokenizer:
    is_fast = True
    pad_token_id = 0

    def __call__(self, text: str, **_: object) -> _FakeEncoding:
        assert text.endswith("[normalized]")
        return _FakeEncoding(
            input_ids=[1, 10, 11, 2],
            attention_mask=[1, 1, 1, 1],
        )


def test_multiban_adapter_and_preflight_use_the_same_normalized_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_normalize(text: str, **_: object) -> str:
        return " ".join(text.split()) + " [normalized]"

    module = types.ModuleType("normalizer")
    module.normalize = fake_normalize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "normalizer", module)
    config = load_yaml_config("config/benchmark_multibanfakedetect.yaml")
    config["dataset"]["root"] = str(tmp_path)
    config["dataset"]["artifacts_dir"] = str(tmp_path / "artifacts")
    rows = {
        "train": ("Train.csv", "Train", "train-image"),
        "val": ("Validation.csv", "Validation", "val-image"),
        "test": ("Test.csv", "Test", "test-image"),
    }
    for _, (csv_name, image_dir, image_id) in rows.items():
        directory = tmp_path / image_dir
        directory.mkdir()
        Image.new("RGB", (8, 8), color="white").save(directory / f"{image_id}.png")
        pd.DataFrame(
            [
                {
                    "image_id": image_id,
                    "headline": "বাংলা   শিরোনাম",
                    "description": "সংবাদ বিবরণ",
                    "category": "national",
                    "types of fake news": "non-fake",
                    "label": 0,
                }
            ]
        ).to_csv(tmp_path / csv_name, index=False)

    dataset = MultiBanFakeDetectDataset(
        config,
        split="train",
        tokenizer=_FakeFastTokenizer(),
    )
    item = dataset[0]
    report_path = prepare_binary_benchmark(config)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert item["text"] == "বাংলা শিরোনাম সংবাদ বিবরণ [normalized]"
    assert item["metadata"]["text_normalizer"] == BANGLA_NORMALIZER_PROVIDER
    assert report_path.name == "benchmark_multibanfakedetect_compact_v1_audit.json"
    assert report["rows"] == 3
    assert report["text_normalization"]["changed_rows"] == 3
    assert report["text_normalization"]["empty_rows"] == 0
