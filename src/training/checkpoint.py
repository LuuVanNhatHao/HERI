"""Checkpoint save/load and retention for Phase-2 baselines."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.training.state import TrainingState
from src.training.optimizer import ensure_no_uninitialized_parameters


def validate_baseline_checkpoint_config(
    current_config: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject legacy/cross-variant baseline checkpoints before loading weights."""

    current_type = current_config.get("model", {}).get("baseline_type")
    checkpoint_type = checkpoint_config.get("model", {}).get("baseline_type")
    if checkpoint_type != current_type:
        raise ValueError(
            f"{context} baseline checkpoint mismatch: current={current_type!r}, "
            f"checkpoint={checkpoint_type!r}"
        )
    current_version = current_config.get("project", {}).get("architecture_version")
    checkpoint_version = checkpoint_config.get("project", {}).get("architecture_version")
    if current_version != "baseline_dense_bbox_v1" or checkpoint_version != current_version:
        raise ValueError(
            f"{context} baseline architecture mismatch: current={current_version!r}, "
            f"checkpoint={checkpoint_version!r}; final baselines require "
            "'baseline_dense_bbox_v1'"
        )


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
    scheduler_plan: Mapping[str, Any] | None = None,
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
            "best_score": list(state.best_score) if state.best_score is not None else None,
            "best_checkpoint": str(state.best_checkpoint) if state.best_checkpoint is not None else None,
            "early_stopping_best_metric": state.early_stopping_best_metric,
            "early_stopping_bad_epochs": state.early_stopping_bad_epochs,
            "thresholds": state.thresholds.as_dict(),
            "metrics": dict(metrics),
            "config": dict(config),
            "precision": precision,
            "scheduler_plan": dict(scheduler_plan) if scheduler_plan is not None else None,
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
    return restore_checkpoint(
        checkpoint=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        restore_rng=restore_rng,
    )


def restore_checkpoint(
    *,
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore an already-loaded checkpoint without a second disk read."""

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
