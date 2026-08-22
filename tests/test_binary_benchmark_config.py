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


def test_multiban_uses_all_compact_backbones_and_isolates_the_new_run() -> None:
    config = load_yaml_config(CONFIGS[0])
    validate_binary_benchmark_config(config, mode="prepare")

    assert config["tokenizer"]["name"] == "csebuetnlp/banglabert_small"
    assert config["backbones"]["siglip2"]["name"] == "google/siglip2-base-patch16-384"
    assert config["backbones"]["dinov3"]["name"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert config["backbones"]["deberta"]["name"] == "csebuetnlp/banglabert_small"
    assert config["dataset"]["text"]["normalization"] == {
        "enabled": True,
        "provider": "csebuetnlp/normalizer",
        "unicode_norm": "NFKC",
        "punct_replacement": None,
        "url_replacement": None,
        "emoji_replacement": None,
        "apply_unicode_norm_last": True,
    }
    assert config["output"]["root"].endswith("multibanfakedetect_compact_v1")
    assert config["project"]["config_name"] == "benchmark_multibanfakedetect_compact_v1"
    assert config["checkpointing"]["minimum_stage"] == "B"
    assert config["training"]["epochs"] == 8
    assert config["training"]["micro_batch_size"] == 16
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["dataloader"]["batch_size"] == 16
    assert config["training"]["early_stopping"]["patience"] == 2
    assert config["training"]["early_stopping"]["minimum_backbone_adaptation_epochs"] == 1


@pytest.mark.parametrize(
    ("key", "large_name"),
    (
        ("siglip2", "google/siglip2-so400m-patch14-384"),
        ("dinov3", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
        ("deberta", "csebuetnlp/banglabert"),
    ),
)
def test_multiban_rejects_noncompact_backbone(key: str, large_name: str) -> None:
    config = copy.deepcopy(load_yaml_config(CONFIGS[0]))
    config["backbones"][key]["name"] = large_name
    if key == "deberta":
        config["tokenizer"]["name"] = large_name

    with pytest.raises(ConfigError, match="requires the compact backbone"):
        validate_binary_benchmark_config(config, mode="prepare")


def test_multiban_rejects_missing_banglabert_normalization() -> None:
    config = copy.deepcopy(load_yaml_config(CONFIGS[0]))
    config["dataset"]["text"]["normalization"]["enabled"] = False

    with pytest.raises(ConfigError, match="requires text normalization"):
        validate_binary_benchmark_config(config, mode="prepare")


def test_vldbench_uses_modernbert_and_isolates_long_context_run() -> None:
    config = load_yaml_config(CONFIGS[1])
    validate_binary_benchmark_config(config, mode="prepare")

    assert config["project"]["architecture_version"] == "hera_binary_long_context_v3"
    assert config["project"]["config_name"] == "benchmark_vldbench_modernbert_long_v1"
    assert config["tokenizer"]["name"] == "answerdotai/ModernBERT-large"
    assert config["tokenizer"]["max_length"] == 2048
    assert config["backbones"]["deberta"]["name"] == "answerdotai/ModernBERT-large"
    assert config["dataset"]["text"]["long_context"] == {
        "enabled": True,
        "strategy": "headline_lead_tail_token_budget_v1",
        "headline_max_tokens": 128,
        "article_lead_fraction": 0.67,
        "audit_batch_size": 128,
    }
    assert config["training"]["micro_batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["dataloader"]["batch_size"] == 1
    assert config["output"]["root"].endswith("vldbench_modernbert_long_v1")


def test_vldbench_rejects_legacy_short_context_backbone() -> None:
    config = copy.deepcopy(load_yaml_config(CONFIGS[1]))
    config["backbones"]["deberta"]["name"] = "microsoft/deberta-v3-large"
    config["tokenizer"]["name"] = "microsoft/deberta-v3-large"

    with pytest.raises(ConfigError, match="ModernBERT-large"):
        validate_binary_benchmark_config(config, mode="prepare")
