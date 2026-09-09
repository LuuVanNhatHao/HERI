from __future__ import annotations

import copy

import pytest

from src.utils.config import (
    ConfigError,
    load_yaml_config,
    validate_baseline_config,
    validate_evidence_config,
    validate_hera_config,
)


def _without_model_dtype(path: str) -> dict:
    config = copy.deepcopy(load_yaml_config(path))
    config["runtime"].pop("model_dtype")
    return config


def _without_cuda_memory_logging(path: str) -> dict:
    config = copy.deepcopy(load_yaml_config(path))
    config["runtime"].pop("cuda_memory_logging")
    return config


def _without_auto_resume(path: str) -> dict:
    config = copy.deepcopy(load_yaml_config(path))
    config["training"].pop("auto_resume")
    return config


def _without_early_stopping(path: str) -> dict:
    config = copy.deepcopy(load_yaml_config(path))
    config["training"].pop("early_stopping")
    return config


def _without_resume_horizon_policy(path: str) -> dict:
    config = copy.deepcopy(load_yaml_config(path))
    config["scheduler"].pop("resume_horizon_policy")
    return config


def _without_evaluation_checkpoint_resolution(path: str) -> dict:
    config = copy.deepcopy(load_yaml_config(path))
    config["evaluation"].pop("checkpoint_resolution")
    return config


def test_baseline_runtime_model_dtype_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="runtime.model_dtype"):
        validate_baseline_config(_without_model_dtype("config/datasets/dgm4/baselines/b0/train.yaml"), mode="train")


def test_evidence_runtime_model_dtype_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="runtime.model_dtype"):
        validate_evidence_config(_without_model_dtype("config/diagnostics/smoke/evidence.yaml"), mode="smoke")


def test_hera_runtime_model_dtype_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="runtime.model_dtype"):
        validate_hera_config(_without_model_dtype("config/datasets/dgm4/hera/train.yaml"), mode="train")


def test_amp_rejects_low_precision_parameter_storage() -> None:
    config = copy.deepcopy(load_yaml_config("config/datasets/dgm4/hera/train.yaml"))
    config["runtime"]["model_dtype"] = "bf16"
    with pytest.raises(ConfigError, match="model_dtype must be fp32"):
        validate_hera_config(config, mode="train")


def test_hera_cuda_memory_logging_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="runtime.cuda_memory_logging.enabled"):
        validate_hera_config(_without_cuda_memory_logging("config/datasets/dgm4/hera/train.yaml"), mode="train")


def test_hera_auto_resume_config_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="training.auto_resume.enabled"):
        validate_hera_config(_without_auto_resume("config/datasets/dgm4/hera/train.yaml"), mode="train")


def test_hera_early_stopping_config_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="training.early_stopping.enabled"):
        validate_hera_config(_without_early_stopping("config/datasets/dgm4/hera/train.yaml"), mode="train")


def test_hera_resume_horizon_policy_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="scheduler.resume_horizon_policy"):
        validate_hera_config(_without_resume_horizon_policy("config/datasets/dgm4/hera/train.yaml"), mode="train")


def test_hera_evaluation_checkpoint_resolution_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="evaluation.checkpoint_resolution.enabled"):
        validate_hera_config(_without_evaluation_checkpoint_resolution("config/datasets/dgm4/hera/evaluate.yaml"), mode="evaluate")
