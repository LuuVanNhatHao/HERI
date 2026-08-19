"""Real-data audits and preparation for standalone binary benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from .data.multiban import IMAGE_EXTENSIONS
from .splits import build_vldbench_split_manifest


def prepare_binary_benchmark(
    config: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Audit the configured release and create required deterministic artifacts."""

    adapter = str(config["dataset"]["adapter"]).strip().casefold()
    if adapter == "multibanfakedetect":
        report = audit_multibanfakedetect(config)
    elif adapter == "vldbench":
        _, manifest_metadata = build_vldbench_split_manifest(config, force=force)
        report = audit_vldbench(config)
        report["split_manifest"] = manifest_metadata
    else:
        raise ValueError(f"Unknown binary benchmark adapter: {adapter!r}")
    output_dir = Path(
        str(config["dataset"].get("artifacts_dir", "outputs/benchmark_preflight"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{adapter}_audit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def audit_multibanfakedetect(config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit every released row/image and report cross-split duplicates."""

    dataset = config["dataset"]
    root = Path(str(dataset["root"]))
    frames: list[pd.DataFrame] = []
    file_hash_rows: list[dict[str, Any]] = []
    split_report: dict[str, Any] = {}
    for split, prefix in (("train", "train"), ("val", "val"), ("test", "test")):
        frame = pd.read_csv(root / str(dataset[f"{prefix}_csv"]))
        frame = frame.copy()
        frame["split"] = split
        frames.append(frame)
        image_root = root / str(dataset[f"{prefix}_images"])
        images = {
            path.stem: path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        }
        wanted = set(frame["image_id"].astype(str))
        missing = sorted(wanted - set(images))
        if missing:
            raise FileNotFoundError(
                f"MultiBanFakeDetect {split} missing images: {missing[:5]}"
            )
        formats: Counter[str] = Counter()
        for sample_id in frame["image_id"].astype(str):
            path = images[sample_id]
            raw = path.read_bytes()
            with Image.open(path) as image:
                image.load()
                formats[str(image.format)] += 1
            file_hash_rows.append(
                {
                    "split": split,
                    "image_id": sample_id,
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        split_report[split] = {
            "rows": int(len(frame)),
            "labels": _json_counts(frame["label"]),
            "subtypes": _json_counts(
                frame["types of fake news"].astype(str).str.strip().str.casefold()
            ),
            "image_formats": dict(formats),
            "missing_images": 0,
        }
    combined = pd.concat(frames, ignore_index=True)
    hashes = pd.DataFrame(file_hash_rows)
    combined = combined.merge(hashes, on=["split", "image_id"], validate="one_to_one")
    combined["text_key"] = (
        combined["headline"].map(_normalize_text)
        + "\u241f"
        + combined["description"].map(_normalize_text)
    )
    combined["full_key"] = combined["text_key"] + "\u241f" + combined["image_sha256"]
    full_groups = combined.groupby("full_key", sort=False)
    conflicting_full = [
        group[["split", "image_id", "label"]].to_dict("records")
        for _, group in full_groups
        if len(group) > 1 and group["label"].nunique() > 1
    ]
    image_groups = combined.groupby("image_sha256", sort=False)
    return {
        "dataset": "MultiBanFakeDetect",
        "rows": int(len(combined)),
        "labels": _json_counts(combined["label"]),
        "splits": split_report,
        "null_required_values": int(
            combined[
                [
                    "image_id",
                    "headline",
                    "description",
                    "category",
                    "types of fake news",
                    "label",
                ]
            ]
            .isna()
            .sum()
            .sum()
        ),
        "unique_image_hashes": int(combined["image_sha256"].nunique()),
        "full_decode_failures": 0,
        "duplicate_image_excess": int(
            len(combined) - combined["image_sha256"].nunique()
        ),
        "cross_split_image_groups": int(
            sum(group["split"].nunique() > 1 for _, group in image_groups)
        ),
        "cross_split_image_label_conflict_groups": int(
            sum(
                group["split"].nunique() > 1 and group["label"].nunique() > 1
                for _, group in image_groups
            )
        ),
        "exact_multimodal_duplicate_groups": int(
            sum(len(group) > 1 for _, group in full_groups)
        ),
        "conflicting_exact_multimodal_groups": conflicting_full,
        "normalizations": {
            "category": {"busniness": "business", "national ": "national"},
            "subtype": {"Clickbait": "clickbait"},
        },
    }


def audit_vldbench(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check the complete public VLDBench Parquet and generated manifest."""

    dataset = config["dataset"]
    parquet_path = Path(str(dataset["parquet"]))
    manifest_path = Path(str(dataset["split_manifest"]["path"]))
    parquet = pq.ParquetFile(parquet_path)
    required_columns = [
        "unique_id",
        "headline",
        "article_text",
        "image",
        "text_label",
        "multimodal_label",
        "news_category",
    ]
    required_table = pq.read_table(
        parquet_path,
        columns=required_columns,
        memory_map=True,
    )
    manifest = pd.read_csv(manifest_path)
    if parquet.metadata.num_rows != len(manifest):
        raise ValueError(
            "VLDBench manifest row count does not match the Parquet release"
        )
    if manifest["unique_id"].nunique() != len(manifest):
        raise ValueError("VLDBench manifest unique_id values are not unique")
    group_splits = manifest.groupby("image_sha256")["split"].nunique()
    image_formats: Counter[str] = Counter()
    for index, image_object in enumerate(required_table["image"].to_pylist()):
        image_bytes = None if image_object is None else image_object.get("bytes")
        if not image_bytes:
            raise ValueError(f"VLDBench image bytes are missing at source_row={index}")
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_formats[str(image.format)] += 1
    return {
        "dataset": "VLDBench",
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.metadata.num_row_groups),
        "parquet_bytes": int(parquet_path.stat().st_size),
        "null_required_values": int(
            sum(required_table[name].null_count for name in required_columns)
        ),
        "image_formats": dict(image_formats),
        "full_decode_failures": 0,
        "schema": [str(field) for field in parquet.schema_arrow],
        "split_counts": _json_counts(manifest["split"]),
        "label_by_split": pd.crosstab(
            manifest["split"], manifest["multimodal_label"]
        ).to_dict(),
        "text_label_by_split": pd.crosstab(
            manifest["split"], manifest["text_label"]
        ).to_dict(),
        "category_by_split": pd.crosstab(
            manifest["split"], manifest["news_category"]
        ).to_dict(),
        "unique_image_hashes": int(manifest["image_sha256"].nunique()),
        "cross_split_image_groups": int((group_splits > 1).sum()),
    }


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _json_counts(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value) for key, value in values.value_counts(dropna=False).items()
    }
