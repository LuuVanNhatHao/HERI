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
    best_checkpoint: Path | None = None
    thresholds: Thresholds = field(default_factory=lambda: DEFAULT_THRESHOLDS)
