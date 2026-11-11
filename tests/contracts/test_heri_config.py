from __future__ import annotations

import pytest

from heri.utils.config import ConfigError, load_yaml_config, validate_hera_config


def test_heri_yaml_configs_validate() -> None:
    files = {
        "config/datasets/dgm4/heri/train.yaml": "train",
        "config/datasets/dgm4/heri/evaluate.yaml": "evaluate",
    }
    for path, mode in files.items():
        config = load_yaml_config(path)
        validate_hera_config(config, mode=mode)
        assert config["dataset"]["root"] == "data/DGM4"
        assert config["model"]["model_type"] == "hera"
        assert config["model"]["binary"]["aggregation"] == "noisy_or"
        assert config["model"]["grounding"]["text"]["aggregation"] == "relative_soft"
        assert "lambda_ev" in config["loss"]
        stages = config["training"]["stages"]
        assert stages["stage_b_full_unfreeze"]["enabled"] is True
        assert stages["stage_b_full_unfreeze"]["start_epoch"] == stages["stage_a_freeze_epochs"] + 1
        assert "stage_b_unfreeze_last_blocks" not in stages


def test_canonical_hera_rejects_unknown_text_grounding_aggregation() -> None:
    config = load_yaml_config("config/datasets/dgm4/heri/train.yaml")
    config["model"]["grounding"]["text"]["aggregation"] = "oracle"
    with pytest.raises(ConfigError, match="text.aggregation"):
        validate_hera_config(config, mode="train")


def test_canonical_hera_rejects_disabled_or_delayed_full_finetuning() -> None:
    config = load_yaml_config("config/datasets/dgm4/heri/train.yaml")
    config["training"]["stages"]["stage_b_full_unfreeze"]["enabled"] = False
    with pytest.raises(ConfigError, match="requires full-backbone fine-tuning"):
        validate_hera_config(config, mode="train")

    config = load_yaml_config("config/datasets/dgm4/heri/train.yaml")
    config["training"]["stages"]["stage_b_full_unfreeze"]["start_epoch"] = 5
    with pytest.raises(ConfigError, match="must immediately follow"):
        validate_hera_config(config, mode="train")


def test_canonical_training_uses_validated_physical_batch_sixteen() -> None:
    config = load_yaml_config("config/datasets/dgm4/heri/train.yaml")

    assert config["training"]["micro_batch_size"] == 16
    assert config["dataloader"]["batch_size"] == 16
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert (
        config["dataloader"]["batch_size"] * config["training"]["gradient_accumulation_steps"] == 16
    )
