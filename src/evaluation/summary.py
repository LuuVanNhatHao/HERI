"""summary.csv and prediction artifact writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.evaluation.evaluator import PredictionBundle
from src.thresholding.tuner import Thresholds


def write_predictions_csv(
    *,
    bundle: PredictionBundle,
    thresholds: Thresholds,
    path: Path,
) -> None:
    """Persist per-sample predictions for later subgroup analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, sample_id in enumerate(bundle.ids):
        atomic_scores = bundle.atomic_scores[idx]
        atomic_targets = bundle.atomic_targets[idx]
        rows.append(
            {
                "id": sample_id,
                "fake_cls": bundle.fake_cls[idx],
                "binary_target": bundle.binary_targets[idx],
                "binary_probability": bundle.binary_scores[idx],
                "predicted_binary": int(bundle.binary_scores[idx] >= thresholds.binary),
                "fs_target": atomic_targets[0],
                "fa_target": atomic_targets[1],
                "ts_target": atomic_targets[2],
                "ta_target": atomic_targets[3],
                "fs_probability": atomic_scores[0],
                "fa_probability": atomic_scores[1],
                "ts_probability": atomic_scores[2],
                "ta_probability": atomic_scores[3],
                "predicted_fs": int(atomic_scores[0] >= thresholds.atomic[0]),
                "predicted_fa": int(atomic_scores[1] >= thresholds.atomic[1]),
                "predicted_ts": int(atomic_scores[2] >= thresholds.atomic[2]),
                "predicted_ta": int(atomic_scores[3] >= thresholds.atomic[3]),
                "generator": bundle.generator[idx] or "",
                "image_source": _value_or_blank(bundle.image_source, idx),
                "annotation_image_path": _value_or_blank(bundle.annotation_image_path, idx),
                "has_image_manipulation": bundle.has_image_manipulation[idx],
                "has_text_manipulation": bundle.has_text_manipulation[idx],
                "face_area_fraction": _value_or_blank(bundle.face_area_fraction, idx),
                "face_size_bucket": _value_or_blank(bundle.face_size_bucket, idx),
                "image_grounding_pred_index": _value_or_blank(bundle.image_grounding_pred_indices, idx),
                "image_grounding_iou": _value_or_blank(bundle.image_grounding_pred_ious, idx),
                "image_grounding_pred_box": _value_or_blank(bundle.image_grounding_pred_boxes, idx),
                "text_grounding_target_word_ids": _value_or_blank(bundle.text_target_word_ids, idx),
                "text_grounding_pred_word_ids": _value_or_blank(bundle.text_pred_word_ids, idx),
                "text_word_probabilities": _value_or_blank(bundle.text_word_probabilities, idx),
                **(bundle.text_sample_counts[idx] if idx < len(bundle.text_sample_counts) else {}),
                "router_diagnostics": _value_or_blank(bundle.router_diagnostics, idx),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary_csv(
    *,
    path: Path,
    run_info: Mapping[str, Any],
    metrics: Mapping[str, Any],
    thresholds: Thresholds,
    reports: Mapping[str, Any],
) -> None:
    """Write one pandas-compatible summary row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    binary_report = reports["binary_report_dict"]
    attr_report = reports["attribution_report_dict"]
    row: dict[str, Any] = {
        **run_info,
        **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))},
        **thresholds.as_dict(),
        "auc": metrics.get("auc"),
        "acc": metrics.get("acc"),
        "eer": metrics.get("eer"),
        "map": metrics.get("map"),
        "cf1": metrics.get("cf1"),
        "of1": metrics.get("of1"),
        "image_mean_iou": metrics.get("image_mean_iou"),
        "image_iou50": metrics.get("image_iou50"),
        "image_iou75": metrics.get("image_iou75"),
        "text_precision": metrics.get("text_precision"),
        "text_recall": metrics.get("text_recall"),
        "text_f1": metrics.get("text_f1"),
        "real_precision": binary_report["real"]["precision"],
        "real_recall": binary_report["real"]["recall"],
        "real_f1": binary_report["real"]["f1-score"],
        "fake_precision": binary_report["fake"]["precision"],
        "fake_recall": binary_report["fake"]["recall"],
        "fake_f1": binary_report["fake"]["f1-score"],
        "fs_precision": attr_report["face_swap"]["precision"],
        "fs_recall": attr_report["face_swap"]["recall"],
        "fs_f1": attr_report["face_swap"]["f1-score"],
        "fa_precision": attr_report["face_attribute"]["precision"],
        "fa_recall": attr_report["face_attribute"]["recall"],
        "fa_f1": attr_report["face_attribute"]["f1-score"],
        "ts_precision": attr_report["text_swap"]["precision"],
        "ts_recall": attr_report["text_swap"]["recall"],
        "ts_f1": attr_report["text_swap"]["f1-score"],
        "ta_precision": attr_report["text_attribute"]["precision"],
        "ta_recall": attr_report["text_attribute"]["recall"],
        "ta_f1": attr_report["text_attribute"]["f1-score"],
    }
    pd.DataFrame([row]).to_csv(path, index=False)


def _value_or_blank(values: list[Any], idx: int) -> Any:
    if idx >= len(values):
        return ""
    value = values[idx]
    return "" if value is None else value
