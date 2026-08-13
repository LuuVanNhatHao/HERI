from __future__ import annotations

import pytest
import torch

from src.training.scheduler import (
    LINEAR_DECAY_CONTINUATION,
    create_continuation_scheduler_plan,
    create_scheduler_from_plan,
)


def _config() -> dict:
    return {
        "scheduler": {
            "type": "linear_warmup",
            "warmup_steps": 500,
            "resume_horizon_policy": "continue_from_checkpoint_lr",
        }
    }


def _step(optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
    optimizer.step()
    scheduler.step()


def test_continuation_scheduler_starts_at_checkpoint_lr_and_decays_to_zero() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.03)
    optimizer.param_groups[0]["initial_lr"] = 0.1
    plan = create_continuation_scheduler_plan(
        _config(),
        remaining_steps=4,
        source_total_epochs=10,
        resume_epoch=10,
        target_total_epochs=20,
        start_global_step=65_060,
        anchor_lrs=[0.03],
    )

    scheduler = create_scheduler_from_plan(optimizer, plan, anchor_current_lrs=True)

    assert scheduler is not None
    assert plan["type"] == LINEAR_DECAY_CONTINUATION
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.03)
    _step(optimizer, scheduler)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0225)
    for _ in range(3):
        _step(optimizer, scheduler)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_continuation_scheduler_plan_survives_crash_resume_without_lr_jump() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.03)
    plan = create_continuation_scheduler_plan(
        _config(),
        remaining_steps=10,
        source_total_epochs=10,
        resume_epoch=10,
        target_total_epochs=20,
        start_global_step=65_060,
        anchor_lrs=[0.03],
    )
    scheduler = create_scheduler_from_plan(optimizer, plan, anchor_current_lrs=True)
    assert scheduler is not None
    for _ in range(3):
        _step(optimizer, scheduler)
    saved_lr = float(optimizer.param_groups[0]["lr"])

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=0.1)
    resumed_scheduler = create_scheduler_from_plan(resumed_optimizer, plan)
    assert resumed_scheduler is not None
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed_scheduler.load_state_dict(scheduler.state_dict())

    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr)
    _step(resumed_optimizer, resumed_scheduler)
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.018)
    assert resumed_optimizer.param_groups[0]["lr"] < saved_lr
