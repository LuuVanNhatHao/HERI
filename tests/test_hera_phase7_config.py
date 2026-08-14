from __future__ import annotations

import pytest

from src.utils.config import ConfigError, load_yaml_config, validate_hera_config, validate_server_validation_config


def test_phase7_hera_yaml_configs_validate() -> None:
    files = {
        "config/train_hera.yaml": "train",
        "config/evaluate_hera.yaml": "evaluate",
        "config/smoke_hera.yaml": "smoke",
    }
    for path, mode in files.items():
        config = load_yaml_config(path)
        validate_hera_config(config, mode=mode)
        assert config["dataset"]["root"] == "data/DGM4"
        assert config["model"]["model_type"] == "hera"
        assert config["model"]["binary"]["aggregation"] == "noisy_or"
        assert "lambda_ev" in config["loss"]
        stages = config["training"]["stages"]
        assert stages["stage_b_full_unfreeze"]["enabled"] is True
        assert stages["stage_b_full_unfreeze"]["start_epoch"] == stages["stage_a_freeze_epochs"] + 1
        assert "stage_b_unfreeze_last_blocks" not in stages


def test_canonical_hera_rejects_disabled_or_delayed_full_finetuning() -> None:
    config = load_yaml_config("config/train_hera.yaml")
    config["training"]["stages"]["stage_b_full_unfreeze"]["enabled"] = False
    with pytest.raises(ConfigError, match="requires full-backbone fine-tuning"):
        validate_hera_config(config, mode="train")

    config = load_yaml_config("config/train_hera.yaml")
    config["training"]["stages"]["stage_b_full_unfreeze"]["start_epoch"] = 5
    with pytest.raises(ConfigError, match="must immediately follow"):
        validate_hera_config(config, mode="train")


def test_server_validation_config_uses_canonical_python_config_commands() -> None:
    config = load_yaml_config("config/server_validation.yaml")
    validate_server_validation_config(config)
    assert config["output"]["root"] == "outputs/server_validation"
    assert config["server"] == {
        "expected_gpu_vram_gb": 32,
        "expected_disk_gb": 150,
        "cuda_environment": "13.2",
    }
    for phase in config["validation"]["phases"]:
        for check in phase["checks"]:
            command = check["command"]
            assert command[0] == "python"
            assert command[1].startswith("scripts/")
            assert command[2].startswith("config/")


def test_server_validation_finishes_with_full_finetune_capacity_gate() -> None:
    config = load_yaml_config("config/server_validation.yaml")
    final_check = config["validation"]["phases"][-1]["checks"][-1]

    assert final_check["name"] == "phase7_full_finetune_capacity"
    assert final_check["required_for_training"] is True
    assert final_check["command"] == [
        "python",
        "scripts/full_ft_capacity_test.py",
        "config/train_hera.yaml",
        "--batch-size",
        "4",
        "--batches",
        "4",
        "--gradient-accumulation-steps",
        "2",
    ]
    assert final_check["environment"] == {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True"
    }


def test_canonical_training_uses_safe_physical_batch_with_effective_batch_eight() -> None:
    config = load_yaml_config("config/train_hera.yaml")

    assert config["training"]["micro_batch_size"] == 4
    assert config["dataloader"]["batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert (
        config["dataloader"]["batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        == 8
    )
