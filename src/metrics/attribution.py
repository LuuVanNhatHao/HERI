"""Fine-grained multilabel attribution metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def attribution_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    thresholds: np.ndarray,
) -> dict[str, float | None]:
    """Compute mAP, CF1, and OF1 for [FS, FA, TS, TA]."""

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.asarray(thresholds).astype(float)
    y_pred = (y_score >= thresholds.reshape(1, -1)).astype(int)

    ap_values: list[float] = []
    for class_idx in range(y_true.shape[1]):
        if len(np.unique(y_true[:, class_idx])) < 2:
            continue
        ap_values.append(float(average_precision_score(y_true[:, class_idx], y_score[:, class_idx])))
    mean_ap = float(np.mean(ap_values)) if ap_values else None
    return {
        "map": mean_ap,
        "cf1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "of1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
