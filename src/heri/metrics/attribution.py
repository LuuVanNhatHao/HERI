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
    """Compute mAP, CF1, and OF1 for an arbitrary ``[N,K]`` ontology."""

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thresholds = np.asarray(thresholds).astype(float)
    y_pred = (y_score >= thresholds.reshape(1, -1)).astype(int)

    ap_values: list[float] = []
    for class_idx in range(y_true.shape[1]):
        if not bool(y_true[:, class_idx].any()):
            ap_values.append(0.0)
        else:
            ap_values.append(float(average_precision_score(y_true[:, class_idx], y_score[:, class_idx])))
    mean_ap = float(np.mean(ap_values)) if ap_values else None
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0)
    predicted = (y_pred == 1).sum(axis=0)
    positives = (y_true == 1).sum(axis=0)
    # DGM4/HAMMER CF1 is H(mean precision, mean recall), NOT mean F1.
    cp = float(np.mean(tp / np.maximum(predicted, 1)))
    cr = float(np.mean(tp / np.maximum(positives, 1)))
    cf1 = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
    return {
        "map": mean_ap,
        "cf1": cf1,
        "class_precision": cp,
        "class_recall": cr,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "of1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
