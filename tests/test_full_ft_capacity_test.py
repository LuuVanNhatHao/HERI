from __future__ import annotations

import pytest

from scripts.full_ft_capacity_test import _resolve_cuda_device, build_capacity_config
from src.utils.config import load_yaml_config, validate_hera_config


def test_capacity_config_forces_immediate_full_finetuning_without_resume() -> None:
    config = build_capacity_config(
        load_yaml_config("config/train_hera.yaml"),
        batch_size=4,
        batches=4,
        gradient_accumulation_steps=2,
    )

    assert config["training"]["micro_batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert config["dataloader"]["batch_size"] == 4
    assert config["training"]["stages"] == {
        "stage_a_freeze_epochs": 0,
        "stage_b_full_unfreeze": {"enabled": True, "start_epoch": 1},
    }
    assert config["training"]["auto_resume"]["enabled"] is False
    assert config["training"]["resume_from"] is None
    assert config["runtime"]["gradient_checkpointing"]["enabled"] is True
    assert config["runtime"]["gradient_checkpointing"]["use_reentrant"] is False
    validate_hera_config(config, mode="train")


@pytest.mark.parametrize(
    "batch_size,batches,accumulation",
    [(0, 2, 1), (8, 0, 1), (-1, 1, 1), (4, 2, 0)],
)
def test_capacity_config_rejects_nonpositive_work(
    batch_size: int,
    batches: int,
    accumulation: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_capacity_config(
            load_yaml_config("config/train_hera.yaml"),
            batch_size=batch_size,
            batches=batches,
            gradient_accumulation_steps=accumulation,
        )


def test_capacity_config_requires_complete_accumulation_windows() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        build_capacity_config(
            load_yaml_config("config/train_hera.yaml"),
            batch_size=4,
            batches=3,
            gradient_accumulation_steps=2,
        )


def test_capacity_test_resolves_bare_cuda_to_current_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.full_ft_capacity_test.torch.cuda.current_device", lambda: 0)

    assert str(_resolve_cuda_device("cuda")) == "cuda:0"
    assert str(_resolve_cuda_device("cuda:2")) == "cuda:2"
