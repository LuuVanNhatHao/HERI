"""Binary detection metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def binary_detection_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | None]:
    """Compute AUC, ACC, and EER with edge-case handling."""

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float | None] = {"acc": float(accuracy_score(y_true, y_pred))}
    if len(np.unique(y_true)) < 2:
        metrics["auc"] = None
        metrics["eer"] = None
        return metrics
    metrics["auc"] = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    metrics["eer"] = float((fpr[idx] + fnr[idx]) * 0.5)
    return metrics
