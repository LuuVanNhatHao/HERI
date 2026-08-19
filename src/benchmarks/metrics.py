"""Binary-first metrics, threshold tuning, and reports for new benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BinaryPredictionBundle:
    ids: list[str] = field(default_factory=list)
    targets: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    subtype_targets: list[int] = field(default_factory=list)
    subtype_predictions: list[int] = field(default_factory=list)
    text_label_targets: list[float] = field(default_factory=list)
    text_label_scores: list[float] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    loss_sums: dict[str, float] = field(default_factory=dict)
    batches: int = 0

    def add_losses(self, values: Mapping[str, float]) -> None:
        for key, value in values.items():
            self.loss_sums[key] = self.loss_sums.get(key, 0.0) + float(value)
        self.batches += 1

    def mean_losses(self) -> dict[str, float]:
        denominator = max(1, self.batches)
        return {key: value / denominator for key, value in self.loss_sums.items()}


def tune_binary_threshold(
    targets: list[float] | np.ndarray,
    scores: list[float] | np.ndarray,
    config: Mapping[str, Any],
) -> float:
    """Tune one validation-only threshold using binary macro F1."""

    y_true = np.asarray(targets, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    settings = config["binary"]
    minimum = float(settings["min"])
    maximum = float(settings["max"])
    step = float(settings["step"])
    if not (0 <= minimum <= maximum <= 1) or step <= 0:
        raise ValueError("Invalid binary threshold search range")
    best_threshold = 0.5
    best_key = (-np.inf, -np.inf, -np.inf)
    for threshold in np.arange(minimum, maximum + step / 2, step):
        predictions = (y_score >= threshold).astype(np.int64)
        macro = f1_score(y_true, predictions, average="macro", zero_division=0)
        positive = f1_score(y_true, predictions, pos_label=1, zero_division=0)
        # Prefer a threshold closer to 0.5 only after both F1 objectives tie.
        key = (float(macro), float(positive), -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics_and_report(
    bundle: BinaryPredictionBundle,
    *,
    threshold: float,
    negative_name: str,
    positive_name: str,
) -> tuple[dict[str, float], str]:
    y_true = np.asarray(bundle.targets, dtype=np.int64)
    y_score = np.asarray(bundle.scores, dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("Cannot compute binary metrics for an empty prediction bundle")
    y_pred = (y_score >= float(threshold)).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": _safe_auc(y_true, y_score),
        "average_precision": _safe_average_precision(y_true, y_score),
        "binary_threshold": float(threshold),
        **{f"loss_{key}": float(value) for key, value in bundle.mean_losses().items()},
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[negative_name, positive_name],
        digits=4,
        zero_division=0,
    )
    return metrics, report


def auxiliary_metrics(bundle: BinaryPredictionBundle) -> dict[str, float]:
    """Report only dataset-native auxiliary targets that were actually predicted."""

    metrics: dict[str, float] = {}
    subtype_target = np.asarray(bundle.subtype_targets, dtype=np.int64)
    subtype_prediction = np.asarray(bundle.subtype_predictions, dtype=np.int64)
    subtype_valid = (subtype_target >= 0) & (subtype_prediction >= 0)
    if bool(subtype_valid.any()):
        metrics.update(
            {
                "subtype_samples": float(subtype_valid.sum()),
                "subtype_accuracy": float(
                    accuracy_score(
                        subtype_target[subtype_valid], subtype_prediction[subtype_valid]
                    )
                ),
                "subtype_macro_f1": float(
                    f1_score(
                        subtype_target[subtype_valid],
                        subtype_prediction[subtype_valid],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    text_target = np.asarray(bundle.text_label_targets, dtype=np.float64)
    text_score = np.asarray(bundle.text_label_scores, dtype=np.float64)
    text_valid = np.isfinite(text_target) & np.isfinite(text_score)
    if bool(text_valid.any()):
        targets = text_target[text_valid].astype(np.int64)
        scores = text_score[text_valid]
        predictions = (scores >= 0.5).astype(np.int64)
        metrics.update(
            {
                "text_label_samples": float(text_valid.sum()),
                "text_label_accuracy_at_0_5": float(
                    accuracy_score(targets, predictions)
                ),
                "text_label_macro_f1_at_0_5": float(
                    f1_score(targets, predictions, average="macro", zero_division=0)
                ),
                "text_label_roc_auc": _safe_auc(targets, scores),
                "text_label_average_precision": _safe_average_precision(
                    targets, scores
                ),
            }
        )
    return metrics


def _safe_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(targets, scores))
        if np.unique(targets).size == 2
        else float("nan")
    )


def _safe_average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(average_precision_score(targets, scores))
        if np.unique(targets).size == 2
        else float("nan")
    )
