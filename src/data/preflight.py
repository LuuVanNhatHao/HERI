"""DGM4 Phase-1 preflight validation and artifact generation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
from tqdm.auto import tqdm

from src.utils.config import dump_yaml, get_key, require_key
from src.utils.environment import collect_environment_metadata
from src.utils.hashing import sha256_file

from .bbox import BBoxError, best_iou_match, validate_box, validate_boxes
from .constants import (
    EXPECTED_ANNOTATION_SHA256,
    EXPECTED_FAKE_BOX_MTCNN_EXACT_MATCHES,
    EXPECTED_MANIPULATED_IMAGE_ROWS,
    EXPECTED_SPLIT_ROWS,
    EXPECTED_TEXT_MANIPULATED_ROWS,
    EXPECTED_TOTAL_ROWS,
    LABEL_ORDER,
)
from .dataset import load_annotations
from .labels import (
    atomic_counts,
    atomic_target,
    binary_target,
    has_image_manipulation,
    has_text_manipulation,
    modality_group,
)
from .paths import resolve_image_path
from .schema import DGM4SchemaError, validate_row, validate_text_positions


def _progress(config: Mapping[str, Any]) -> bool:
    return bool(get_key(config, "progress.enabled", True))


def _split_annotation_path(config: Mapping[str, Any], split: str) -> Path:
    key = {"train": "train_annotations", "val": "val_annotations", "test": "test_annotations"}[split]
    return Path(str(require_key(config, f"dataset.{key}")))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _read_image_header_or_decode(item: tuple[str, str], *, full_decode: bool) -> tuple[str, dict[str, Any]]:
    annotation_path, physical_path = item
    result: dict[str, Any] = {
        "ok": False,
        "width": None,
        "height": None,
        "format": None,
        "mode": None,
        "error": None,
    }
    try:
        with Image.open(physical_path) as image:
            result["width"], result["height"] = image.size
            result["format"] = image.format
            result["mode"] = image.mode
            if full_decode:
                image.load()
        result["ok"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    return annotation_path, result


class PreflightRecorder:
    """Collects violations and compact examples during preflight."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, Any]]] = {}

    def add(self, key: str, example: dict[str, Any] | None = None) -> None:
        self.counts[key] += 1
        if example is not None:
            bucket = self.examples.setdefault(key, [])
            if len(bucket) < 20:
                bucket.append(example)

    def has_failures(self) -> bool:
        return any(count > 0 for count in self.counts.values())


def run_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run configured DGM4 preflight and write Phase-1 artifacts."""

    progress = _progress(config)
    output_root = Path(str(require_key(config, "output.root")))
    output_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_root / "effective_config.yaml")

    checks = {
        "verify_hashes": bool(get_key(config, "dataset.verify_hashes", True)),
        "verify_schema": bool(require_key(config, "preflight.verify_schema")),
        "verify_paths": bool(require_key(config, "preflight.verify_paths")),
        "verify_image_headers": bool(require_key(config, "preflight.verify_image_headers")),
        "verify_full_decode": bool(require_key(config, "preflight.verify_full_decode")),
        "verify_boxes": bool(require_key(config, "preflight.verify_boxes")),
        "verify_mtcnn_match": bool(require_key(config, "preflight.verify_mtcnn_match")),
        "verify_text_positions": bool(require_key(config, "preflight.verify_text_positions")),
        "verify_split_counts": bool(require_key(config, "preflight.verify_split_counts")),
    }
    tqdm.write(f"Enabled preflight checks: {checks}")

    environment = collect_environment_metadata(Path.cwd())
    (output_root / "environment.json").write_text(
        json.dumps(environment, indent=2, default=_json_default),
        encoding="utf-8",
    )

    recorder = PreflightRecorder()
    dataset_root = Path(str(require_key(config, "dataset.root")))
    image_root = Path(str(require_key(config, "dataset.image_root")))
    splits = ("train", "val", "test")
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, dict[str, Any]] = {}

    if checks["verify_hashes"]:
        for split in splits:
            path = _split_annotation_path(config, split)
            observed = sha256_file(path, progress=progress)
            expected = EXPECTED_ANNOTATION_SHA256[split]
            hashes[split] = {
                "path": str(path),
                "expected": expected,
                "observed": observed,
                "ok": observed == expected,
            }
            if observed != expected:
                recorder.add(
                    "hash_mismatch",
                    {"split": split, "path": str(path), "expected": expected, "observed": observed},
                )
    else:
        hashes = {split: {"enabled": False} for split in splits}

    for split in tqdm(
        splits,
        total=len(splits),
        desc="Reading DGM4 annotations",
        disable=not progress,
    ):
        rows_by_split[split] = load_annotations(_split_annotation_path(config, split), split=split)

    split_counts = {split: len(rows) for split, rows in rows_by_split.items()}
    total_rows = sum(split_counts.values())
    if checks["verify_split_counts"]:
        for split, expected in EXPECTED_SPLIT_ROWS.items():
            if split_counts[split] != expected:
                recorder.add(
                    "split_count_mismatch",
                    {"split": split, "expected": expected, "observed": split_counts[split]},
                )
        if total_rows != EXPECTED_TOTAL_ROWS:
            recorder.add("total_count_mismatch", {"expected": EXPECTED_TOTAL_ROWS, "observed": total_rows})

    label_counts: dict[str, dict[str, int]] = {}
    modality_counts: dict[str, dict[str, int]] = {}
    manifest_rows: list[dict[str, Any]] = []
    resolved_paths: dict[str, str] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    fake_image_rows = 0
    text_manipulated_rows = 0
    mtcnn_exact_matches = 0

    for split, rows in rows_by_split.items():
        iterator = tqdm(
            enumerate(rows),
            total=len(rows),
            desc=f"DGM4 {split} schema/paths",
            disable=not progress,
        )
        label_counter: Counter[str] = Counter()
        modality_counter: Counter[str] = Counter()
        for index, row in iterator:
            if checks["verify_schema"]:
                try:
                    validate_row(row, split=split, row_index=index)
                except DGM4SchemaError as exc:
                    recorder.add(
                        "schema_error",
                        {"split": split, "row_index": index, "error": str(exc)},
                    )
                    continue
            label_counter[row["fake_cls"]] += 1
            modality_counter[modality_group(row["fake_cls"])] += 1
            try:
                resolved = resolve_image_path(
                    row["image"],
                    dataset_root,
                    image_root=image_root,
                    verify_exists=checks["verify_paths"],
                )
                resolved_paths[row["image"]] = str(resolved.physical_path)
            except Exception as exc:
                recorder.add(
                    "path_error",
                    {
                        "split": split,
                        "row_index": index,
                        "id": row.get("id"),
                        "image": row.get("image"),
                        "error": str(exc),
                    },
                )
                continue
            atomic = atomic_target(row["fake_cls"])
            manifest_rows.append(
                {
                    "id": row["id"],
                    "split": split,
                    "fake_cls": row["fake_cls"],
                    "binary_target": binary_target(row["fake_cls"]),
                    "fs": atomic[0],
                    "fa": atomic[1],
                    "ts": atomic[2],
                    "ta": atomic[3],
                    "annotation_image_path": row["image"],
                    "resolved_image_path": str(resolved.physical_path),
                    "image_source": resolved.source or "",
                    "generator": resolved.generator or "",
                    "has_image_manipulation": int(has_image_manipulation(row["fake_cls"])),
                    "has_text_manipulation": int(has_text_manipulation(row["fake_cls"])),
                    "num_faces": len(row["mtcnn_boxes"]),
                    "num_fake_words": len(row["fake_text_pos"]),
                }
            )
        label_counts[split] = dict(label_counter)
        modality_counts[split] = dict(modality_counter)

    if checks["verify_image_headers"] or checks["verify_full_decode"]:
        unique_items = list(resolved_paths.items())
        worker_threads = int(get_key(config, "preflight.worker_threads", 8))
        with ThreadPoolExecutor(max_workers=max(1, worker_threads)) as executor:
            futures = [
                executor.submit(
                    _read_image_header_or_decode,
                    item,
                    full_decode=checks["verify_full_decode"],
                )
                for item in unique_items
            ]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="DGM4 image decode" if checks["verify_full_decode"] else "DGM4 image headers",
                disable=not progress,
            )
            for future in iterator:
                annotation_path, result = future.result()
                if not result["ok"]:
                    recorder.add(
                        "image_decode_error" if checks["verify_full_decode"] else "image_header_error",
                        {
                            "annotation_image_path": annotation_path,
                            "resolved_image_path": resolved_paths.get(annotation_path),
                            "error": result["error"],
                        },
                    )
                    continue
                image_sizes[annotation_path] = (int(result["width"]), int(result["height"]))

    if checks["verify_boxes"] or checks["verify_mtcnn_match"] or checks["verify_text_positions"]:
        for split, rows in rows_by_split.items():
            iterator = tqdm(
                enumerate(rows),
                total=len(rows),
                desc=f"DGM4 {split} boxes/text",
                disable=not progress,
            )
            for index, row in iterator:
                fake_cls = row["fake_cls"]
                image_size = image_sizes.get(row["image"])
                if checks["verify_boxes"] and image_size is None:
                    recorder.add(
                        "missing_image_size_for_box_check",
                        {"split": split, "row_index": index, "image": row["image"]},
                    )
                    continue
                if checks["verify_boxes"]:
                    try:
                        validate_boxes(
                            row["mtcnn_boxes"],
                            image_size=image_size,
                            field_name="mtcnn_boxes",
                        )
                        if has_image_manipulation(fake_cls):
                            validate_box(
                                row["fake_image_box"],
                                image_size=image_size,
                                field_name="fake_image_box",
                            )
                    except BBoxError as exc:
                        recorder.add(
                            "bbox_error",
                            {
                                "split": split,
                                "row_index": index,
                                "id": row["id"],
                                "image": row["image"],
                                "error": str(exc),
                            },
                        )
                if has_image_manipulation(fake_cls):
                    fake_image_rows += 1
                    if checks["verify_mtcnn_match"]:
                        try:
                            best_index, best_iou = best_iou_match(
                                row["fake_image_box"],
                                row["mtcnn_boxes"],
                            )
                        except BBoxError as exc:
                            recorder.add(
                                "mtcnn_match_error",
                                {
                                    "split": split,
                                    "row_index": index,
                                    "id": row["id"],
                                    "error": str(exc),
                                },
                            )
                            continue
                        if best_index >= 0 and abs(best_iou - 1.0) <= 1e-9:
                            mtcnn_exact_matches += 1
                        else:
                            recorder.add(
                                "mtcnn_non_exact_match",
                                {
                                    "split": split,
                                    "row_index": index,
                                    "id": row["id"],
                                    "best_index": best_index,
                                    "best_iou": best_iou,
                                },
                            )
                if has_text_manipulation(fake_cls):
                    text_manipulated_rows += 1
                if checks["verify_text_positions"]:
                    try:
                        validate_text_positions(
                            row["text"].split(),
                            row["fake_text_pos"],
                            context=f"split={split} index={index} id={row['id']}",
                        )
                    except DGM4SchemaError as exc:
                        recorder.add(
                            "text_position_error",
                            {
                                "split": split,
                                "row_index": index,
                                "id": row["id"],
                                "error": str(exc),
                            },
                        )

    if (
        checks["verify_split_counts"]
        and checks["verify_boxes"]
        and fake_image_rows != EXPECTED_MANIPULATED_IMAGE_ROWS
    ):
        recorder.add(
            "manipulated_image_row_count_mismatch",
            {"expected": EXPECTED_MANIPULATED_IMAGE_ROWS, "observed": fake_image_rows},
        )
    if (
        checks["verify_split_counts"]
        and checks["verify_text_positions"]
        and text_manipulated_rows != EXPECTED_TEXT_MANIPULATED_ROWS
    ):
        recorder.add(
            "text_manipulated_row_count_mismatch",
            {"expected": EXPECTED_TEXT_MANIPULATED_ROWS, "observed": text_manipulated_rows},
        )
    if (
        checks["verify_split_counts"]
        and checks["verify_mtcnn_match"]
        and mtcnn_exact_matches != EXPECTED_FAKE_BOX_MTCNN_EXACT_MATCHES
    ):
        recorder.add(
            "mtcnn_exact_match_count_mismatch",
            {"expected": EXPECTED_FAKE_BOX_MTCNN_EXACT_MATCHES, "observed": mtcnn_exact_matches},
        )

    split_overlaps: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_ids = {row["id"] for row in rows_by_split[left]}
        right_ids = {row["id"] for row in rows_by_split[right]}
        left_images = {row["image"] for row in rows_by_split[left]}
        right_images = {row["image"] for row in rows_by_split[right]}
        split_overlaps[f"{left}-{right}"] = {
            "id_overlap": len(left_ids & right_ids),
            "image_overlap": len(left_images & right_images),
        }
        if split_overlaps[f"{left}-{right}"]["id_overlap"]:
            recorder.add("cross_split_id_overlap", split_overlaps[f"{left}-{right}"])
        if split_overlaps[f"{left}-{right}"]["image_overlap"]:
            recorder.add("cross_split_image_overlap", split_overlaps[f"{left}-{right}"])

    total_label_counts = Counter()
    total_modality_counts = Counter()
    for split in rows_by_split:
        total_label_counts.update(label_counts[split])
        total_modality_counts.update(modality_counts[split])
    total_atomic_counts = atomic_counts(
        row["fake_cls"] for rows in rows_by_split.values() for row in rows
    )

    if bool(get_key(config, "preflight.write_manifest", True)):
        manifest_path = output_root / "dataset_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            iterator = tqdm(
                manifest_rows,
                total=len(manifest_rows),
                desc="Writing manifest",
                disable=not progress,
            )
            for row in iterator:
                writer.writerow(row)
    else:
        manifest_path = None

    disabled_checks = sorted(check for check, enabled in checks.items() if not enabled)
    status = "FAIL" if recorder.has_failures() else (
        "PASS_WITH_DISABLED_CHECKS" if disabled_checks else "PASS"
    )
    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": status,
        "checks": checks,
        "disabled_checks": disabled_checks,
        "hashes": hashes,
        "split_counts": split_counts,
        "total_rows": total_rows,
        "label_counts": label_counts,
        "total_label_counts": dict(total_label_counts),
        "atomic_counts": dict(total_atomic_counts),
        "modality_counts": modality_counts,
        "total_modality_counts": dict(total_modality_counts),
        "path_validation": {
            "unique_resolved_images": len(resolved_paths),
            "errors": int(recorder.counts.get("path_error", 0)),
        },
        "image_validation": {
            "unique_images_checked": len(image_sizes),
            "header_or_decode_errors": int(
                recorder.counts.get("image_header_error", 0)
                + recorder.counts.get("image_decode_error", 0)
            ),
            "full_decode_enabled": checks["verify_full_decode"],
        },
        "bbox_validation": {
            "manipulated_image_rows": fake_image_rows,
            "bbox_errors": int(recorder.counts.get("bbox_error", 0)),
        },
        "text_position_validation": {
            "text_manipulated_rows": text_manipulated_rows,
            "errors": int(recorder.counts.get("text_position_error", 0)),
        },
        "mtcnn_matching": {
            "exact_matches": mtcnn_exact_matches,
            "expected_exact_matches": EXPECTED_FAKE_BOX_MTCNN_EXACT_MATCHES,
            "errors": int(
                recorder.counts.get("mtcnn_match_error", 0)
                + recorder.counts.get("mtcnn_non_exact_match", 0)
            ),
        },
        "split_overlaps": split_overlaps,
        "violations": {
            "counts": dict(recorder.counts),
            "examples": recorder.examples,
        },
        "artifacts": {
            "dataset_manifest_csv": str(manifest_path) if manifest_path else None,
            "environment_json": str(output_root / "environment.json"),
            "effective_config_yaml": str(output_root / "effective_config.yaml"),
        },
    }
    (output_root / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return audit
