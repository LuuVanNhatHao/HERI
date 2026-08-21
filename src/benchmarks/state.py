"""Serializable training state for binary benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class BenchmarkTrainingState:
    epoch: int = 0
    global_step: int = 0
    threshold: float = 0.5
    best_metric: float | None = None
    best_threshold: float | None = None
    best_model_path: Path | None = None
    early_stopping_best: float | None = None
    early_stopping_bad_epochs: int = 0

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "BenchmarkTrainingState":
        return cls(
            epoch=int(payload["epoch"]),
            global_step=int(payload["global_step"]),
            threshold=float(payload.get("binary_threshold", 0.5)),
            best_metric=(
                float(payload["best_metric"])
                if payload.get("best_metric") is not None
                else None
            ),
            best_threshold=(
                float(payload["best_binary_threshold"])
                if payload.get("best_binary_threshold") is not None
                else None
            ),
            best_model_path=(
                Path(str(payload["best_model_path"]))
                if payload.get("best_model_path") not in (None, "")
                else None
            ),
            early_stopping_best=(
                float(payload["early_stopping_best_metric"])
                if payload.get("early_stopping_best_metric") is not None
                else None
            ),
            early_stopping_bad_epochs=int(payload.get("early_stopping_bad_epochs", 0)),
        )

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "binary_threshold": float(self.threshold),
            "best_metric": self.best_metric,
            "best_binary_threshold": self.best_threshold,
            "best_model_path": str(self.best_model_path)
            if self.best_model_path
            else None,
            "early_stopping_best_metric": self.early_stopping_best,
            "early_stopping_bad_epochs": int(self.early_stopping_bad_epochs),
        }
