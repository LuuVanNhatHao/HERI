"""Deterministic, image-group-safe split construction for VLDBench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold


def build_vldbench_split_manifest(
    config: Mapping[str, Any],
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Create a reproducible 70/15/15-style split without crossing image hashes."""

    dataset = config["dataset"]
    parquet_path = Path(str(dataset["parquet"]))
    split_config = dataset["split_manifest"]
    manifest_path = Path(str(split_config["path"]))
    metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".metadata.json")
    if manifest_path.exists() and not force:
        if not metadata_path.is_file():
            raise ValueError(
                f"Existing VLDBench manifest has no metadata: {metadata_path}. "
                "Rerun preparation with --force."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_split_config = _split_config_signature(split_config)
        if metadata.get("split_config") != expected_split_config:
            raise ValueError(
                "Existing VLDBench manifest was built with a different split contract. "
                "Rerun preparation with --force."
            )
        if metadata.get("manifest_sha256") != _sha256_file(manifest_path):
            raise ValueError(
                "Existing VLDBench manifest hash does not match its metadata"
            )
        if metadata.get("source_sha256") != _sha256_file(parquet_path):
            raise ValueError(
                "VLDBench Parquet changed after the split manifest was created"
            )
        metadata = {**metadata, "status": "verified_existing"}
        return manifest_path, metadata
    if not parquet_path.is_file():
        raise FileNotFoundError(f"VLDBench parquet does not exist: {parquet_path}")

    table = pq.read_table(
        parquet_path,
        columns=[
            "unique_id",
            "multimodal_label",
            "text_label",
            "news_category",
            "image",
        ],
        memory_map=True,
    )
    unique_ids = np.asarray(table["unique_id"].to_pylist(), dtype=object)
    labels = np.asarray(table["multimodal_label"].to_pylist(), dtype=object)
    text_labels = np.asarray(table["text_label"].to_pylist(), dtype=object)
    categories = np.asarray(table["news_category"].to_pylist(), dtype=object)
    image_hashes = np.asarray(
        [
            hashlib.sha256(value["bytes"]).hexdigest()
            for value in table["image"].to_pylist()
        ],
        dtype=object,
    )
    if len(set(unique_ids.tolist())) != len(unique_ids):
        raise ValueError("VLDBench unique_id values are not unique")

    order = np.argsort(unique_ids.astype(str), kind="stable")
    strata = np.char.add(
        np.char.add(labels[order].astype(str), "|"), categories[order].astype(str)
    )
    groups = image_hashes[order]
    folds = int(split_config["folds"])
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=int(split_config["seed"]),
    )
    fold_by_sorted_row = np.full(len(order), -1, dtype=np.int16)
    dummy = np.zeros(len(order), dtype=np.int8)
    for fold, (_, held_out) in enumerate(splitter.split(dummy, strata, groups)):
        fold_by_sorted_row[held_out] = fold
    if bool((fold_by_sorted_row < 0).any()):
        raise RuntimeError("VLDBench fold assignment left unassigned rows")

    fold_by_source_row = np.empty(len(order), dtype=np.int16)
    fold_by_source_row[order] = fold_by_sorted_row
    train_folds = {int(value) for value in split_config["train_folds"]}
    val_folds = {int(value) for value in split_config["val_folds"]}
    test_folds = {int(value) for value in split_config["test_folds"]}
    expected_folds = set(range(folds))
    if train_folds | val_folds | test_folds != expected_folds:
        raise ValueError(
            "VLDBench train/val/test folds must cover every configured fold"
        )
    if train_folds & val_folds or train_folds & test_folds or val_folds & test_folds:
        raise ValueError("VLDBench train/val/test fold sets must be disjoint")

    def split_name(fold: int) -> str:
        if fold in train_folds:
            return "train"
        if fold in val_folds:
            return "val"
        return "test"

    manifest = pd.DataFrame(
        {
            "unique_id": unique_ids.astype(str),
            "source_row": np.arange(len(unique_ids), dtype=np.int64),
            "split": [split_name(int(fold)) for fold in fold_by_source_row],
            "fold": fold_by_source_row,
            "multimodal_label": labels.astype(str),
            "text_label": text_labels.astype(str),
            "news_category": categories.astype(str),
            "image_sha256": image_hashes.astype(str),
        }
    )
    _assert_no_group_leakage(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(manifest_path)

    metadata = {
        "status": "created",
        "manifest": str(manifest_path),
        "source_parquet": str(parquet_path),
        "source_sha256": _sha256_file(parquet_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "seed": int(split_config["seed"]),
        "folds": folds,
        "split_config": _split_config_signature(split_config),
        "split_counts": manifest["split"].value_counts().sort_index().to_dict(),
        "label_by_split": pd.crosstab(
            manifest["split"], manifest["multimodal_label"]
        ).to_dict(),
        "category_by_split": pd.crosstab(
            manifest["split"], manifest["news_category"]
        ).to_dict(),
        "unique_image_hashes": int(manifest["image_sha256"].nunique()),
        "image_group_overlap": 0,
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return manifest_path, metadata


def _split_config_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(config["seed"]),
        "folds": int(config["folds"]),
        "train_folds": [int(value) for value in config["train_folds"]],
        "val_folds": [int(value) for value in config["val_folds"]],
        "test_folds": [int(value) for value in config["test_folds"]],
    }


def _assert_no_group_leakage(manifest: pd.DataFrame) -> None:
    split_counts = manifest.groupby("image_sha256")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if len(leaking):
        raise RuntimeError(
            f"VLDBench image hashes cross split boundaries: {len(leaking)} groups"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
