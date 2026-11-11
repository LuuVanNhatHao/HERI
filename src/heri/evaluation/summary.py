"""summary.csv and prediction artifact writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from heri.evaluation.evaluator import (
    PredictionBundle,
    exclusive_class_decision_scores,
    gated_image_iou_for_sample,
)
from heri.evaluation.samm_protocol import (
    SAMM_RELEASED_LABEL_ORDER,
    is_samm_reporting_ontology,
    samm_released_scores,
    samm_released_targets,
)
from heri.thresholding.tuner import Thresholds


def _column_token(name: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")


def write_predictions_csv(
    *,
    bundle: PredictionBundle,
    thresholds: Thresholds,
    path: Path,
) -> None:
    """Persist per-sample predictions for later subgroup analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    hypothesis_names = list(
        bundle.hypothesis_names or thresholds.hypothesis_names
    )
    if len(hypothesis_names) != len(thresholds.atomic):
        raise ValueError("Prediction hypothesis names and thresholds do not align")
    samm_protocol = is_samm_reporting_ontology(bundle.fake_cls, hypothesis_names)
    official_targets = (
        samm_released_targets(bundle.fake_cls) if samm_protocol else None
    )
    official_scores = (
        samm_released_scores(
            np.asarray(bundle.atomic_scores, dtype=float),
            np.asarray(bundle.binary_scores, dtype=float),
            hypothesis_names,
        )
        if samm_protocol
        else None
    )
    for idx, sample_id in enumerate(bundle.ids):
        atomic_scores = bundle.atomic_scores[idx]
        atomic_targets = bundle.atomic_targets[idx]
        evidence_support = (
            bundle.evidence_support_scores[idx]
            if idx < len(bundle.evidence_support_scores)
            else None
        )
        modality_support = (
            bundle.evidence_support_by_modality_scores[idx]
            if idx < len(bundle.evidence_support_by_modality_scores)
            else None
        )
        presence_score = _value_or_blank(bundle.image_presence_scores, idx)
        predicts_image_presence = (
            int(float(presence_score) >= float(thresholds.image_presence))
            if presence_score != "" and thresholds.image_presence is not None
            else ""
        )
        ungated_box = _value_or_blank(bundle.image_grounding_pred_boxes, idx)
        gated_box = (
            "[0.0, 0.0, 0.0, 0.0]"
            if predicts_image_presence == 0
            else ungated_box
        )
        row = {
                "id": sample_id,
                "fake_cls": bundle.fake_cls[idx],
                "binary_target": bundle.binary_targets[idx],
                "binary_probability": bundle.binary_scores[idx],
                "predicted_binary": int(bundle.binary_scores[idx] >= thresholds.binary),
                "generator": _value_or_blank(bundle.generator, idx),
                "image_source": _value_or_blank(bundle.image_source, idx),
                "annotation_image_path": _value_or_blank(bundle.annotation_image_path, idx),
                "has_image_manipulation": bundle.has_image_manipulation[idx],
                "has_text_manipulation": bundle.has_text_manipulation[idx],
                "image_grounding_available": _value_or_blank(
                    bundle.image_grounding_available, idx
                ),
                "text_grounding_available": _value_or_blank(
                    bundle.text_grounding_available, idx
                ),
                "face_area_fraction": _value_or_blank(bundle.face_area_fraction, idx),
                "face_size_bucket": _value_or_blank(bundle.face_size_bucket, idx),
                "image_grounding_pred_index": _value_or_blank(bundle.image_grounding_pred_indices, idx),
                "image_presence_probability": presence_score,
                "image_presence_threshold": thresholds.image_presence,
                "predicted_image_presence": predicts_image_presence,
                "image_grounding_iou": gated_image_iou_for_sample(
                    bundle, idx, thresholds.image_presence
                ),
                "image_grounding_pred_box": gated_box,
                "image_grounding_ungated_pred_box": ungated_box,
                "image_grounding_pred_box_original_pixels": _value_or_blank(bundle.image_grounding_pred_boxes_original, idx),
                "image_grounding_box_frame": "normalized_padded_canvas",
                "text_grounding_target_word_ids": _value_or_blank(bundle.text_target_word_ids, idx),
                "text_grounding_pred_word_ids": _value_or_blank(bundle.text_pred_word_ids, idx),
                "text_word_probabilities": _value_or_blank(bundle.text_word_probabilities, idx),
                "evidence_decision_metrics_available": evidence_support is not None,
                "evidence_decision_threshold": 0.5 if evidence_support is not None else "",
                "evidence_decision_unit": (
                    "sample_hypothesis_modality"
                    if modality_support is not None
                    else ""
                ),
                **(bundle.text_sample_counts[idx] if idx < len(bundle.text_sample_counts) else {}),
                "router_diagnostics": _value_or_blank(bundle.router_diagnostics, idx),
                "hypothesis_probabilities_by_cycle": _value_or_blank(bundle.hypothesis_probabilities_by_cycle, idx),
            }
        for hypothesis_index, name in enumerate(hypothesis_names):
            token = _column_token(name)
            row[f"{token}_target"] = atomic_targets[hypothesis_index]
            row[f"{token}_probability"] = atomic_scores[hypothesis_index]
            row[f"predicted_{token}"] = int(
                atomic_scores[hypothesis_index] >= thresholds.atomic[hypothesis_index]
            )
            if evidence_support is None:
                row[f"{token}_evidence_support"] = ""
                row[f"{token}_evidence_support_joint_min"] = ""
                row[f"{token}_decision_evidence_gap"] = ""
                row[f"{token}_unsupported_decision"] = ""
            else:
                support = float(evidence_support[hypothesis_index])
                active = bool(
                    atomic_scores[hypothesis_index]
                    >= 0.5
                )
                row[f"{token}_evidence_support"] = support
                row[f"{token}_evidence_support_joint_min"] = support
                diagnostic_support = support
                if modality_support is not None:
                    availability_by_name = {
                        "image": bundle.image_grounding_available,
                        "text": bundle.text_grounding_available,
                    }
                    valid_support = [
                        float(modality_support[hypothesis_index][modality_index])
                        for modality_index, modality_name in enumerate(
                            bundle.evidence_support_modality_names
                        )
                        if bool(
                            bundle.evidence_support_modality_mask[
                                hypothesis_index
                            ][modality_index]
                        )
                        and bool(
                            _value_or_blank(
                                availability_by_name.get(modality_name, []), idx
                            )
                        )
                    ]
                    diagnostic_support = min(valid_support) if valid_support else None
                row[f"{token}_decision_evidence_gap"] = (
                    max(
                        0.0,
                        float(atomic_scores[hypothesis_index])
                        - diagnostic_support,
                    )
                    if diagnostic_support is not None
                    else ""
                )
                row[f"{token}_unsupported_decision"] = (
                    int(
                        active
                        and diagnostic_support
                        < 0.5
                    )
                    if diagnostic_support is not None
                    else ""
                )
            if modality_support is not None:
                for modality_index, modality_name in enumerate(
                    bundle.evidence_support_modality_names
                ):
                    modality_token = _column_token(modality_name)
                    applicable = bool(
                        bundle.evidence_support_modality_mask[hypothesis_index][
                            modality_index
                        ]
                    )
                    available_values = {
                        "image": bundle.image_grounding_available,
                        "text": bundle.text_grounding_available,
                    }.get(modality_name, [])
                    available = bool(
                        _value_or_blank(available_values, idx)
                    ) if available_values else False
                    valid = applicable and available
                    value = float(
                        modality_support[hypothesis_index][modality_index]
                    )
                    row[f"{token}_{modality_token}_evidence_applicable"] = int(
                        applicable
                    )
                    row[f"{token}_{modality_token}_evidence_valid"] = int(valid)
                    row[f"{token}_{modality_token}_evidence_support"] = (
                        value if applicable else ""
                    )
                    row[f"{token}_{modality_token}_decision_evidence_gap"] = (
                        max(0.0, float(atomic_scores[hypothesis_index]) - value)
                        if valid
                        else ""
                    )
                    row[f"{token}_{modality_token}_unsupported_decision"] = (
                        int(active and value < 0.5)
                        if valid
                        else ""
                    )
                    if idx < len(bundle.verified_support_eligible):
                        eligible = bool(bundle.verified_support_eligible[idx][hypothesis_index][modality_index])
                        row[f"{token}_{modality_token}_vsr_at_1_eligible"] = int(eligible)
                        row[f"{token}_{modality_token}_vsr_at_1_hit"] = (
                            int(bundle.verified_support_hits[idx][hypothesis_index][modality_index]) if eligible else ""
                        )
        if official_targets is not None and official_scores is not None:
            for label_index, label in enumerate(SAMM_RELEASED_LABEL_ORDER):
                row[f"samm_official_{label}_target"] = int(
                    official_targets[idx, label_index]
                )
                row[f"samm_official_{label}_probability"] = float(
                    official_scores[idx, label_index]
                )
                row[f"samm_official_predicted_{label}"] = int(
                    official_scores[idx, label_index] >= 0.5
                )
        if idx < len(bundle.class_targets) and bundle.class_targets[idx] >= 0:
            row["class_target"] = bundle.class_targets[idx]
            if bundle.class_names and bundle.class_targets[idx] < len(bundle.class_names):
                row["class_name"] = bundle.class_names[bundle.class_targets[idx]]
            if len(bundle.class_names) == len(atomic_scores) + 1:
                class_scores = exclusive_class_decision_scores(
                    np.asarray([atomic_scores], dtype=float)
                )[0]
                class_pred = int(class_scores.argmax())
                row["predicted_class"] = class_pred
                row["predicted_class_name"] = bundle.class_names[class_pred]
                for class_name, score in zip(bundle.class_names, class_scores):
                    row[f"{_column_token(class_name)}_decision_score"] = float(score)
        rows.append(row)
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
        "image_iou95": metrics.get("image_iou95"),
        "image_presence_precision": metrics.get("image_presence_precision"),
        "image_presence_recall": metrics.get("image_presence_recall"),
        "image_presence_f1": metrics.get("image_presence_f1"),
        "text_precision": metrics.get("text_precision"),
        "text_recall": metrics.get("text_recall"),
        "text_f1": metrics.get("text_f1"),
        "real_precision": binary_report["real"]["precision"],
        "real_recall": binary_report["real"]["recall"],
        "real_f1": binary_report["real"]["f1-score"],
        "fake_precision": binary_report["fake"]["precision"],
        "fake_recall": binary_report["fake"]["recall"],
        "fake_f1": binary_report["fake"]["f1-score"],
    }
    hypothesis_names = reports.get("hypothesis_names", list(thresholds.hypothesis_names))
    report_names = reports.get("hypothesis_report_names", hypothesis_names)
    for hypothesis_name, report_name in zip(hypothesis_names, report_names):
        values = attr_report[report_name]
        token = _column_token(hypothesis_name)
        row[f"{token}_precision"] = values["precision"]
        row[f"{token}_recall"] = values["recall"]
        row[f"{token}_f1"] = values["f1-score"]
    class_report = reports.get("class_report_dict")
    if isinstance(class_report, Mapping):
        for class_name, values in class_report.items():
            if class_name in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if not isinstance(values, Mapping):
                continue
            token = _column_token(str(class_name))
            row[f"class_{token}_precision"] = values.get("precision")
            row[f"class_{token}_recall"] = values.get("recall")
            row[f"class_{token}_f1"] = values.get("f1-score")
    official_report = reports.get("samm_official_report_dict")
    if isinstance(official_report, Mapping):
        for label in SAMM_RELEASED_LABEL_ORDER:
            values = official_report.get(label)
            if not isinstance(values, Mapping):
                continue
            row[f"samm_official_{label}_precision"] = values.get("precision")
            row[f"samm_official_{label}_recall"] = values.get("recall")
            row[f"samm_official_{label}_f1"] = values.get("f1-score")
    pd.DataFrame([row]).to_csv(path, index=False)


def _value_or_blank(values: list[Any], idx: int) -> Any:
    if idx >= len(values):
        return ""
    value = values[idx]
    return "" if value is None else value
