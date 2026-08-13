"""Scheduler factories and serializable plans for Transformer fine-tuning."""

from __future__ import annotations

from typing import Any, Mapping

import torch


LINEAR_WARMUP = "linear_warmup"
LINEAR_DECAY_CONTINUATION = "linear_decay_continuation"


def create_scheduler_plan(config: Mapping[str, Any], *, total_steps: int) -> dict[str, Any]:
    """Create the serializable scheduler plan for a fresh training stage."""

    sched_config = config["scheduler"]
    sched_type = str(sched_config["type"]).lower()
    if sched_type == "none":
        return {"type": "none", "total_steps": int(total_steps), "warmup_steps": 0}
    if sched_type != LINEAR_WARMUP:
        raise ValueError(f"Unsupported scheduler type: {sched_type!r}")
    return {
        "type": LINEAR_WARMUP,
        "total_steps": int(total_steps),
        "warmup_steps": int(sched_config["warmup_steps"]),
    }


def create_continuation_scheduler_plan(
    config: Mapping[str, Any],
    *,
    remaining_steps: int,
    source_total_epochs: int,
    resume_epoch: int,
    target_total_epochs: int,
    start_global_step: int,
    anchor_lrs: list[float],
) -> dict[str, Any]:
    """Plan a no-jump linear decay from checkpoint LRs over the extension."""

    sched_type = str(config["scheduler"]["type"]).lower()
    if sched_type == "none":
        plan_type = "none"
    elif sched_type == LINEAR_WARMUP:
        plan_type = LINEAR_DECAY_CONTINUATION
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type!r}")
    return {
        "type": plan_type,
        "total_steps": int(remaining_steps),
        "warmup_steps": 0,
        "source_total_epochs": int(source_total_epochs),
        "resume_epoch": int(resume_epoch),
        "target_total_epochs": int(target_total_epochs),
        "start_global_step": int(start_global_step),
        "anchor_lrs": [float(value) for value in anchor_lrs],
    }


def create_scheduler_from_plan(
    optimizer: torch.optim.Optimizer,
    plan: Mapping[str, Any],
    *,
    anchor_current_lrs: bool = False,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Instantiate a scheduler from a checkpoint-safe serialized plan."""

    sched_type = str(plan["type"]).lower()
    if sched_type == "none":
        return None
    if sched_type not in {LINEAR_WARMUP, LINEAR_DECAY_CONTINUATION}:
        raise ValueError(f"Unsupported scheduler plan type: {sched_type!r}")
    total_steps = int(plan["total_steps"])
    warmup_steps = int(plan["warmup_steps"])
    if total_steps <= 0:
        raise ValueError("Scheduler plan total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("Scheduler plan warmup_steps must be non-negative")
    if anchor_current_lrs:
        for group in optimizer.param_groups:
            group["initial_lr"] = float(group["lr"])

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create the configured scheduler for a fresh training stage."""

    plan = create_scheduler_plan(config, total_steps=total_steps)
    return create_scheduler_from_plan(optimizer, plan)
