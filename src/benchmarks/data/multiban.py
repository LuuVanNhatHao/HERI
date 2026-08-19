"""MultiBanFakeDetect CSV/image adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from PIL import Image

from src.benchmarks.contracts import SUBTYPE_CLASSES, normalize_multiban_subtype

from .base import BinaryBenchmarkDataset, BinaryBenchmarkRecord


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".jfif"})


class MultiBanFakeDetectDataset(BinaryBenchmarkDataset):
    """Official 80/10/10 MultiBanFakeDetect release with explicit image matching."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        split: str,
        tokenizer: Any,
    ) -> None:
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            tokenizer_max_length=int(config["tokenizer"]["max_length"]),
            image_size=int(config["image"]["size"]),
        )
        dataset = config["dataset"]
        root = Path(str(dataset["root"]))
        split_key = {"train": "train", "val": "val", "test": "test"}[split]
        csv_path = _resolve(root, dataset[f"{split_key}_csv"])
        image_root = _resolve(root, dataset[f"{split_key}_images"])
        self.frame = pd.read_csv(csv_path)
        required = {
            "image_id",
            "headline",
            "description",
            "category",
            "types of fake news",
            "label",
        }
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(
                f"MultiBanFakeDetect {split} CSV is missing columns: {missing}"
            )
        if self.frame[list(required)].isna().any().any():
            raise ValueError(
                f"MultiBanFakeDetect {split} contains null required values"
            )
        self.text_fields = tuple(str(value) for value in dataset["text"]["fields"])
        self.separator = str(dataset["text"]["separator"])
        unknown_text_fields = sorted(set(self.text_fields) - set(self.frame.columns))
        if unknown_text_fields:
            raise ValueError(
                f"Unknown MultiBanFakeDetect text fields: {unknown_text_fields}"
            )
        self.image_paths = _index_images(image_root)
        missing_images = sorted(
            set(self.frame["image_id"].astype(str)) - set(self.image_paths)
        )
        if missing_images:
            raise FileNotFoundError(
                f"MultiBanFakeDetect {split} is missing {len(missing_images)} images; "
                f"examples={missing_images[:5]}"
            )

    def __len__(self) -> int:
        return len(self.frame)

    def raw_record(self, index: int) -> BinaryBenchmarkRecord:
        row = self.frame.iloc[int(index)]
        sample_id = str(row["image_id"])
        binary = int(row["label"])
        if binary not in {0, 1}:
            raise ValueError(
                f"Invalid MultiBanFakeDetect binary label: id={sample_id} label={binary}"
            )
        subtype = normalize_multiban_subtype(row["types of fake news"])
        expected_binary = 0 if subtype == "non-fake" else 1
        if expected_binary != binary:
            raise ValueError(
                f"MultiBanFakeDetect subtype/binary conflict: id={sample_id} "
                f"subtype={subtype!r} label={binary}"
            )
        text = self.separator.join(
            str(row[field]).strip() for field in self.text_fields
        )
        path = self.image_paths[sample_id]
        return BinaryBenchmarkRecord(
            sample_id=sample_id,
            text=text,
            image=Image.open(path),
            image_reference=str(path),
            binary_target=float(binary),
            subtype_target=SUBTYPE_CLASSES.index(subtype),
            subtype_available=True,
            metadata={
                "category": _normalize_category(row["category"]),
                "subtype": subtype,
                "source_row": int(index),
            },
        )


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _index_images(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"MultiBanFakeDetect image directory does not exist: {root}"
        )
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate MultiBanFakeDetect image stem: {path.stem}")
        result[path.stem] = path
    return result


def _normalize_category(value: Any) -> str:
    category = str(value).strip().casefold()
    return "business" if category == "busniness" else category
