"""Checkpoint save/load and retention for Phase-2 baselines."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.training.state import TrainingState
from src.training.optimizer import ensure_no_uninitialized_parameters


def rng_state() -> dict[str, Any]:
    """Capture RNG state where practical."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state where available."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    state: TrainingState,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    precision: str,
) -> None:
    """Save a complete resume-capable checkpoint."""

    ensure_no_uninitialized_parameters(model, context="checkpoint save")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "best_checkpoint": str(state.best_checkpoint) if state.best_checkpoint is not None else None,
            "thresholds": state.thresholds.as_dict(),
            "metrics": dict(metrics),
            "config": dict(config),
            "precision": precision,
            "rng_state": rng_state(),
        },
        path,
    )


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load checkpoint and restore all requested training state."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng and "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint
