"""sklearn classification_report wrappers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report


BINARY_TARGET_NAMES = ["real", "fake"]
ATTRIBUTION_TARGET_NAMES = ["face_swap", "face_attribute", "text_swap", "text_attribute"]


def binary_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return sklearn's binary classification report string."""

    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=BINARY_TARGET_NAMES,
        digits=4,
        zero_division=0,
    )


def attribution_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return sklearn's multilabel attribution classification report string."""

    return classification_report(
        y_true,
        y_pred,
        target_names=ATTRIBUTION_TARGET_NAMES,
        digits=4,
        zero_division=0,
    )


def binary_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return sklearn's binary classification report dictionary."""

    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=BINARY_TARGET_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )


def attribution_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return sklearn's attribution classification report dictionary."""

    return classification_report(
        y_true,
        y_pred,
        target_names=ATTRIBUTION_TARGET_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
