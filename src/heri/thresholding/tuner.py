"""Validation-only threshold tuning for baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from heri.graph.schema import HYPOTHESIS_ORDER


@dataclass(frozen=True)
class Thresholds:
    """Binary, atomic, and optional ontology-aware grounding thresholds."""

    binary: float
    atomic: tuple[float, ...]
    hypothesis_names: tuple[str, ...] = HYPOTHESIS_ORDER
    image_presence: float | None = None
    image_presence_source: str | None = None
    decision_source: str | None = None

    def __post_init__(self) -> None:
        if not self.atomic:
            raise ValueError("At least one atomic threshold is required")
        if len(self.atomic) != len(self.hypothesis_names):
            raise ValueError("Atomic thresholds and hypothesis_names must have equal length")
        if not 0.0 <= float(self.binary) <= 1.0:
            raise ValueError("binary threshold must be in [0,1]")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.atomic):
            raise ValueError("atomic thresholds must be in [0,1]")
        if self.image_presence is not None and not 0.0 <= float(self.image_presence) <= 1.0:
            raise ValueError("image presence threshold must be in [0,1]")
        if self.image_presence is None and self.image_presence_source is not None:
            raise ValueError("image_presence_source requires an image presence threshold")
        if len(set(self.hypothesis_names)) != len(self.hypothesis_names):
            raise ValueError("hypothesis_names must be unique")
        if self.decision_source is None:
            inferred = (
                "fixed_0.5"
                if float(self.binary) == 0.5
                and all(float(value) == 0.5 for value in self.atomic)
                else "caller_supplied"
            )
            object.__setattr__(self, "decision_source", inferred)
        if self.decision_source not in {
            "fixed_0.5",
            "validation_tuned",
            "legacy_validation_tuned",
            "caller_supplied",
        }:
            raise ValueError(f"Unknown decision threshold source: {self.decision_source!r}")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "binary_threshold": self.binary,
            "hypothesis_names": list(self.hypothesis_names),
            "atomic_thresholds": list(self.atomic),
            "decision_threshold_source": str(self.decision_source),
        }
        for name, threshold in zip(self.hypothesis_names, self.atomic):
            result[f"{name.lower()}_threshold"] = threshold
        if self.image_presence is not None:
            result["image_presence_threshold"] = self.image_presence
            result["image_presence_threshold_source"] = (
                self.image_presence_source or "validation_tuned"
            )
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
    image_presence_targets: np.ndarray | None = None,
    image_presence_scores: np.ndarray | None = None,
) -> Thresholds:
    """Tune thresholds exclusively on validation predictions."""

    if not bool(config.get("enabled", True)):
        names = tuple(hypothesis_names or HYPOTHESIS_ORDER[: atomic_targets.shape[1]])
        image_section = config.get("image_presence", {})
        image_threshold = 0.5 if bool(image_section.get("enabled", False)) else None
        return Thresholds(
            binary=0.5,
            atomic=(0.5,) * atomic_targets.shape[1],
            hypothesis_names=names,
            image_presence=image_threshold,
            image_presence_source=("fixed_0.5" if image_threshold is not None else None),
            decision_source="fixed_0.5",
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
    image_presence_threshold = None
    image_presence_source = None
    image_section = config.get("image_presence", {})
    if bool(image_section.get("enabled", False)):
        if image_presence_targets is None or image_presence_scores is None:
            raise ValueError(
                "image presence thresholding requires validation targets and scores"
            )
        image_presence_threshold = _tune_binary(
            binary_targets=np.asarray(image_presence_targets, dtype=int),
            binary_scores=np.asarray(image_presence_scores, dtype=float),
            section=image_section,
            progress=progress,
            description="Tune image-presence threshold",
        )
        image_presence_source = "validation_tuned_macro_f1"
    names = tuple(hypothesis_names or HYPOTHESIS_ORDER[: len(atomic_thresholds)])
    if len(names) != len(atomic_thresholds):
        raise ValueError("hypothesis_names must align with atomic targets")
    return Thresholds(
        binary=binary_threshold,
        atomic=tuple(float(v) for v in atomic_thresholds),
        hypothesis_names=names,
        image_presence=image_presence_threshold,
        image_presence_source=image_presence_source,
        decision_source="validation_tuned",
    )


def _tune_binary(
    *,
    binary_targets: np.ndarray,
    binary_scores: np.ndarray,
    section: Mapping[str, Any],
    progress: bool,
    description: str = "Tune binary threshold",
) -> float:
    best_threshold = 0.5
    best_score = -1.0
    thresholds = _grid(section)
    iterator = tqdm(thresholds, desc=description, disable=not progress)
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
