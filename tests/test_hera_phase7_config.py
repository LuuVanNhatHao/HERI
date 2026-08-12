from __future__ import annotations

from src.utils.config import load_yaml_config, validate_hera_config, validate_server_validation_config


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
        assert config["training"]["stages"]["stage_c_full_unfreeze"]["enabled"] is False


def test_server_validation_config_uses_canonical_python_config_commands() -> None:
    config = load_yaml_config("config/server_validation.yaml")
    validate_server_validation_config(config)
    assert config["output"]["root"] == "outputs/server_validation"
    for phase in config["validation"]["phases"]:
        for check in phase["checks"]:
            command = check["command"]
            assert command[0] == "python"
            assert command[1].startswith("scripts/")
            assert command[2].startswith("config/")
