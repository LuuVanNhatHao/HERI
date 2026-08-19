from __future__ import annotations

import copy

import pytest

from src.benchmarks.capacity import build_binary_capacity_config
from src.benchmarks.config import validate_binary_benchmark_config
from src.utils.config import ConfigError, load_yaml_config


CONFIGS = (
    "config/benchmark_multibanfakedetect.yaml",
    "config/benchmark_vldbench.yaml",
)


@pytest.mark.parametrize("path", CONFIGS)
def test_binary_benchmark_yaml_has_no_dgm4_target_heads(path: str) -> None:
    config = load_yaml_config(path)
    validate_binary_benchmark_config(config, mode="prepare")

    assert config["task"]["type"] == "binary_multimodal"
    assert config["task"]["signals"]["binary"] is True
    assert config["task"]["signals"]["attribution"] is False
    assert config["task"]["signals"]["image_grounding"] is False
    assert config["task"]["signals"]["text_grounding"] is False
    assert "attribution" not in config["model"]
    assert "grounding" not in config["model"]
    assert "hypotheses" not in config["model"]
    assert config["model"]["latent_slots"]["roles"] == [
        "visual_slot_1",
        "visual_slot_2",
        "text_slot_1",
        "text_slot_2",
    ]


def test_auxiliary_signal_and_loss_weight_must_be_enabled_together() -> None:
    config = load_yaml_config(CONFIGS[0])
    config = copy.deepcopy(config)
    config["task"]["signals"]["subtype"] = True

    with pytest.raises(ConfigError, match="lambda_subtype"):
        validate_binary_benchmark_config(config, mode="prepare")


def test_capacity_config_is_full_ft_and_never_writes_training_checkpoints() -> None:
    config = load_yaml_config(CONFIGS[1])
    capacity = build_binary_capacity_config(
        config,
        batch_size=2,
        batches=4,
        gradient_accumulation_steps=2,
    )

    assert capacity["training"]["stages"]["stage_a_freeze_epochs"] == 0
    assert capacity["training"]["stages"]["stage_b_full_unfreeze"] == {
        "enabled": True,
        "start_epoch": 1,
    }
    assert capacity["training"]["auto_resume"]["enabled"] is False
    assert capacity["training"]["gradient_accumulation_steps"] == 2
    assert capacity["dataloader"]["batch_size"] == 2
