"""Validation-only threshold tuning for Phase-2 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from src.graph.schema import HYPOTHESIS_ORDER


@dataclass(frozen=True)
class Thresholds:
    """Binary and atomic decision thresholds."""

    binary: float
    atomic: tuple[float, ...]
    hypothesis_names: tuple[str, ...] = HYPOTHESIS_ORDER

    def __post_init__(self) -> None:
        if not self.atomic:
            raise ValueError("At least one atomic threshold is required")
        if len(self.atomic) != len(self.hypothesis_names):
            raise ValueError("Atomic thresholds and hypothesis_names must have equal length")
        if not 0.0 <= float(self.binary) <= 1.0:
            raise ValueError("binary threshold must be in [0,1]")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.atomic):
            raise ValueError("atomic thresholds must be in [0,1]")
        if len(set(self.hypothesis_names)) != len(self.hypothesis_names):
            raise ValueError("hypothesis_names must be unique")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "binary_threshold": self.binary,
            "hypothesis_names": list(self.hypothesis_names),
            "atomic_thresholds": list(self.atomic),
        }
        for name, threshold in zip(self.hypothesis_names, self.atomic):
            result[f"{name.lower()}_threshold"] = threshold
        return result


DEFAULT_THRESHOLDS = Thresholds(binary=0.5, atomic=(0.5, 0.5, 0.5, 0.5))


def default_thresholds_for_hypotheses(
    hypothesis_names: tuple[str, ...] | list[str],
) -> Thresholds:
    """Return ontology-aligned neutral thresholds for fresh trainer state."""

    names = tuple(str(name) for name in hypothesis_names)
    return Thresholds(binary=0.5, atomic=(0.5,) * len(names), hypothesis_names=names)


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
    hypothesis_names: tuple[str, ...] | list[str] | None = None,
) -> Thresholds:
    """Tune thresholds exclusively on validation predictions."""

    if not bool(config.get("enabled", True)):
        names = tuple(hypothesis_names or HYPOTHESIS_ORDER[: atomic_targets.shape[1]])
        return Thresholds(
            binary=0.5, atomic=(0.5,) * atomic_targets.shape[1], hypothesis_names=names
        )
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
    names = tuple(hypothesis_names or HYPOTHESIS_ORDER[: len(atomic_thresholds)])
    if len(names) != len(atomic_thresholds):
        raise ValueError("hypothesis_names must align with atomic targets")
    return Thresholds(
        binary=binary_threshold,
        atomic=tuple(float(v) for v in atomic_thresholds),
        hypothesis_names=names,
    )


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
