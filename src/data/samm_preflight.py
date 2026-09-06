"""Reproducible preflight audit for the SAMM-20K protocol."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
from tqdm.auto import tqdm

from src.utils.config import dump_yaml, get_key, require_key
from src.utils.hashing import sha256_file

from .samm import (
    SAMM_EXPECTED_SHA256,
    SAMM_EXPECTED_SPLIT_ROWS,
    SAMM_LABEL_TO_TARGET,
    SAMMPathError,
    SAMMSchemaError,
    load_samm_annotations,
    resolve_samm_image_path,
    validate_samm_row,
)


def _annotation_path(config: Mapping[str, Any], split: str) -> Path:
    key = {"train": "train_annotations", "val": "val_annotations", "test": "test_annotations"}[split]
    return Path(str(require_key(config, f"dataset.{key}")))


def run_samm_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frozen splits while treating documented annotation gaps as warnings."""

    output_root = Path(str(require_key(config, "output.root")))
    output_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_root / "effective_config.yaml")
    progress = bool(get_key(config, "progress.enabled", True))
    verify_hashes = bool(get_key(config, "dataset.verify_hashes", True))
    verify_paths = bool(get_key(config, "preflight.verify_paths", True))
    verify_headers = bool(get_key(config, "preflight.verify_image_headers", False))
    header_limit = get_key(config, "preflight.image_header_max_samples", None)
    configured_hashes = config.get("dataset_hashes", SAMM_EXPECTED_SHA256)

    errors: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    split_summary: dict[str, Any] = {}
    seen_ids: dict[int, str] = {}
    seen_paths: dict[str, str] = {}
    header_candidates: list[Path] = []

    for split in ("train", "val", "test"):
        path = _annotation_path(config, split)
        if not path.is_file():
            errors["missing_annotation_file"] += 1
            split_summary[split] = {"annotation_path": str(path), "missing": True}
            continue
        digest = sha256_file(path)
        if verify_hashes and digest != str(configured_hashes.get(split, "")):
            errors["annotation_hash_mismatch"] += 1
        rows = load_samm_annotations(path, split=split)
        if bool(get_key(config, "preflight.verify_split_counts", True)) and len(rows) != SAMM_EXPECTED_SPLIT_ROWS[split]:
            errors["split_row_count_mismatch"] += 1
        class_counts: Counter[str] = Counter()
        iterator = tqdm(enumerate(rows), total=len(rows), desc=f"SAMM {split} audit", disable=not progress)
        for index, row in iterator:
            try:
                validate_samm_row(row, split=split, row_index=index)
            except SAMMSchemaError:
                errors["schema_error"] += 1
                continue
            label = str(row["fake_cls"])
            class_counts[label] += 1
            sample_id = int(row["id"])
            annotation_image = str(row["image"])
            if sample_id in seen_ids:
                errors["cross_split_duplicate_id"] += 1
            else:
                seen_ids[sample_id] = split
            if annotation_image in seen_paths:
                errors["cross_split_duplicate_image_path"] += 1
            else:
                seen_paths[annotation_image] = split
            try:
                physical = resolve_samm_image_path(
                    annotation_image,
                    require_key(config, "dataset.root"),
                    image_root=require_key(config, "dataset.image_root"),
                    fake_cls=label,
                    verify_exists=verify_paths,
                )
                if verify_headers and (header_limit is None or len(header_candidates) < int(header_limit)):
                    header_candidates.append(physical)
            except SAMMPathError:
                errors["image_path_error"] += 1
            manipulated = label != "orig"
            box = row.get("fake_image_box", [])
            if manipulated and not box:
                warnings["manipulated_missing_image_box"] += 1
            elif box and any(float(value) < 0 for value in box):
                warnings["box_requires_clipping"] += 1
            words = str(row["text"]).split()
            if not words:
                warnings["empty_caption"] += 1
            positions = row["fake_text_pos"]
            if manipulated and not positions:
                warnings["manipulated_missing_text_positions"] += 1
            if any(position < 0 or position >= len(words) for position in positions):
                warnings["invalid_text_position_row"] += 1
                warnings["invalid_text_position_count"] += sum(
                    position < 0 or position >= len(words) for position in positions
                )
        split_summary[split] = {
            "annotation_path": str(path),
            "sha256": digest,
            "rows": len(rows),
            "class_counts": dict(class_counts),
        }

    if verify_headers:
        for path in tqdm(header_candidates, desc="SAMM image headers", disable=not progress):
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                errors["image_header_error"] += 1

    if errors:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"
    audit = {
        "dataset": "SAMM",
        "protocol": "samm20k_seed777",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": status,
        "root": str(require_key(config, "dataset.root")),
        "hypothesis_order": ["SWAP", "ATTRIBUTE"],
        "label_to_target": {key: list(value) for key, value in SAMM_LABEL_TO_TARGET.items()},
        "splits": split_summary,
        "unique_ids": len(seen_ids),
        "unique_image_paths": len(seen_paths),
        "errors": dict(errors),
        "warnings": dict(warnings),
        "policies": dict(get_key(config, "dataset.annotation_policy", {})),
    }
    (output_root / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit
