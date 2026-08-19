"""Optimizer, scheduler, model, and RNG restoration for benchmark resumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.training.checkpoint import restore_rng_state
from src.training.optimizer import load_compatible_optimizer_state
from src.training.precision import PrecisionManager
from src.training.scheduler import (
    create_continuation_scheduler_plan,
    create_scheduler_from_plan,
    create_scheduler_plan,
)

from .state import BenchmarkTrainingState
from .training_utils import stage_name


@dataclass(frozen=True)
class RestoredRuntime:
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    scheduler_plan: dict[str, Any]
    message: str | None = None


def restore_resume_runtime(
    *,
    model: torch.nn.Module,
    precision: PrecisionManager,
    config: Mapping[str, Any],
    state: BenchmarkTrainingState,
    payload: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    total_steps: int,
    optimizer_steps_per_epoch: int,
    resume_epoch: int,
    target_epoch: int,
) -> RestoredRuntime:
    model.load_state_dict(payload["model"])
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
    checkpoint_epochs = int(payload["config"]["training"]["epochs"])
    crosses_stage = stage_name(config, resume_epoch) != stage_name(config, target_epoch)
    horizon_changed = checkpoint_epochs != total_epochs
    if not crosses_stage and not horizon_changed:
        optimizer.load_state_dict(payload["optimizer"])
        saved_plan = payload.get("scheduler_plan")
        plan = (
            dict(saved_plan)
            if isinstance(saved_plan, Mapping)
            else create_scheduler_plan(config, total_steps=total_steps)
        )
        scheduler = create_scheduler_from_plan(optimizer, plan)
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler") is not None:
            precision.scaler.load_state_dict(payload["scaler"])
        return RestoredRuntime(scheduler=scheduler, scheduler_plan=plan)
    if total_epochs < checkpoint_epochs:
        raise RuntimeError("Cannot resume with a shorter training horizon")
    loaded = load_compatible_optimizer_state(optimizer, payload["optimizer"])
    remaining_steps = max(1, (total_epochs - resume_epoch) * optimizer_steps_per_epoch)
    anchors = [float(group["lr"]) for group in optimizer.param_groups]
    plan = create_continuation_scheduler_plan(
        config,
        remaining_steps=remaining_steps,
        source_total_epochs=checkpoint_epochs,
        resume_epoch=resume_epoch,
        target_total_epochs=total_epochs,
        start_global_step=state.global_step,
        anchor_lrs=anchors,
    )
    scheduler = create_scheduler_from_plan(optimizer, plan, anchor_current_lrs=True)
    return RestoredRuntime(
        scheduler=scheduler,
        scheduler_plan=plan,
        message=f"Resume optimizer migration: preserved_state_for={loaded}, anchor_lrs={anchors}",
    )
