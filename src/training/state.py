"""Training state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.thresholding.tuner import Thresholds, DEFAULT_THRESHOLDS


@dataclass
class TrainingState:
    """Mutable trainer state needed for checkpoint/resume."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    best_score: tuple[float, ...] | None = None
    best_checkpoint: Path | None = None
    early_stopping_best_metric: float | None = None
    early_stopping_bad_epochs: int = 0
    thresholds: Thresholds = field(default_factory=lambda: DEFAULT_THRESHOLDS)
