"""Validation-only threshold tuning for Phase-2 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


@dataclass(frozen=True)
class Thresholds:
    """Binary and atomic decision thresholds."""

    binary: float
    atomic: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, float]:
        return {
            "binary_threshold": self.binary,
            "fs_threshold": self.atomic[0],
            "fa_threshold": self.atomic[1],
            "ts_threshold": self.atomic[2],
            "ta_threshold": self.atomic[3],
        }


DEFAULT_THRESHOLDS = Thresholds(binary=0.5, atomic=(0.5, 0.5, 0.5, 0.5))


def _grid(section: Mapping[str, Any]) -> np.ndarray:
    minimum = float(section["min"])
    maximum = float(section["max"])
    step = float(section["step"])
    if step <= 0 or maximum < minimum:
        raise ValueError(f"Invalid threshold grid: {dict(section)}")
    count = int(round((maximum - minimum) / step)) + 1
    return np.linspace(minimum, minimum + step * (count - 1), count).clip(minimum, maximum)


def tune_thresholds(
    *,
    binary_targets: np.ndarray,
    binary_scores: np.ndarray,
    atomic_targets: np.ndarray,
    atomic_scores: np.ndarray,
    config: Mapping[str, Any],
    progress: bool = True,
) -> Thresholds:
    """Tune thresholds exclusively on validation predictions."""

    if not bool(config.get("enabled", True)):
        return DEFAULT_THRESHOLDS
    binary_threshold = _tune_binary(
        binary_targets=binary_targets,
        binary_scores=binary_scores,
        section=config["binary"],
        progress=progress,
    )
    atomic_thresholds = _tune_atomic(
        atomic_targets=atomic_targets,
        atomic_scores=atomic_scores,
        section=config["attribution"],
        progress=progress,
    )
    return Thresholds(binary=binary_threshold, atomic=tuple(float(v) for v in atomic_thresholds))


def _tune_binary(
    *,
    binary_targets: np.ndarray,
    binary_scores: np.ndarray,
    section: Mapping[str, Any],
    progress: bool,
) -> float:
    best_threshold = 0.5
    best_score = -1.0
    thresholds = _grid(section)
    iterator = tqdm(thresholds, desc="Tune binary threshold", disable=not progress)
    for threshold in iterator:
        pred = (binary_scores >= threshold).astype(int)
        if section.get("objective", "macro_f1") == "macro_f1":
            score = f1_score(binary_targets, pred, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unsupported binary threshold objective: {section.get('objective')!r}")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _tune_atomic(
    *,
    atomic_targets: np.ndarray,
    atomic_scores: np.ndarray,
    section: Mapping[str, Any],
    progress: bool,
) -> np.ndarray:
    if section.get("mode", "per_class") != "per_class":
        raise ValueError(f"Unsupported attribution threshold mode: {section.get('mode')!r}")
    thresholds = _grid(section)
    best = np.full(atomic_targets.shape[1], 0.5, dtype=float)
    for class_idx in range(atomic_targets.shape[1]):
        best_score = -1.0
        iterator = tqdm(
            thresholds,
            desc=f"Tune atomic threshold {class_idx}",
            disable=not progress,
        )
        for threshold in iterator:
            pred = (atomic_scores[:, class_idx] >= threshold).astype(int)
            score = f1_score(
                atomic_targets[:, class_idx],
                pred,
                zero_division=0,
            )
            if score > best_score:
                best_score = float(score)
                best[class_idx] = float(threshold)
    return best
