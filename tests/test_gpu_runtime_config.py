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


def test_baseline_runtime_model_dtype_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="runtime.model_dtype"):
        validate_baseline_config(_without_model_dtype("config/train_baseline_b0.yaml"), mode="train")


def test_evidence_runtime_model_dtype_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="runtime.model_dtype"):
        validate_evidence_config(_without_model_dtype("config/smoke_evidence.yaml"), mode="smoke")


def test_hera_runtime_model_dtype_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="runtime.model_dtype"):
        validate_hera_config(_without_model_dtype("config/train_hera.yaml"), mode="train")


def test_hera_cuda_memory_logging_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="runtime.cuda_memory_logging.enabled"):
        validate_hera_config(_without_cuda_memory_logging("config/train_hera.yaml"), mode="train")


def test_hera_auto_resume_config_is_mandatory() -> None:
    with pytest.raises(ConfigError, match="training.auto_resume.enabled"):
        validate_hera_config(_without_auto_resume("config/train_hera.yaml"), mode="train")
