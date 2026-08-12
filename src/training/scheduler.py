"""Scheduler factory for Transformer fine-tuning."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create configured scheduler."""

    sched_config = config["scheduler"]
    sched_type = str(sched_config["type"]).lower()
    if sched_type == "none":
        return None
    if sched_type != "linear_warmup":
        raise ValueError(f"Unsupported scheduler type: {sched_type!r}")
    warmup_steps = int(sched_config["warmup_steps"])

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
