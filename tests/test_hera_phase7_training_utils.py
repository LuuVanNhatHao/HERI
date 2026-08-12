from __future__ import annotations

import math

import pytest

from hera_phase7_fixtures import FixtureGraphReasoner, TinyBackbone, hera_test_config
from phase2_fixtures import synthetic_batch, synthetic_features
from src.models.hera import HERADGM
from src.training.hera_trainer import _ensure_finite_losses, apply_configured_gradient_checkpointing
from src.training.optimizer import apply_training_stage, create_optimizer, parameter_accounting


def _stage_model() -> HERADGM:
    return HERADGM(
        hera_test_config(),
        siglip2=TinyBackbone(),
        dinov3=TinyBackbone(),
        deberta=TinyBackbone(),
        graph_reasoner=FixtureGraphReasoner(hidden_dim=12),
        load_backbones=False,
    )


class CheckpointableTinyBackbone(TinyBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False


def test_hera_stage_a_freezes_backbones_and_optimizer_has_new_hera_group() -> None:
    model = _stage_model()
    model(synthetic_batch(), features=synthetic_features())

    cfg = hera_test_config()
    apply_training_stage(model, cfg, epoch=1)
    assert all(not p.requires_grad for p in model.siglip2.parameters())
    assert all(not p.requires_grad for p in model.dinov3.parameters())
    assert all(not p.requires_grad for p in model.deberta.parameters())
    _, groups = create_optimizer(model, cfg)
    assert [group.name for group in groups] == ["new_hera_modules"]


def test_hera_stage_b_partial_unfreezes_backbone_optimizer_groups_and_audit_keys() -> None:
    model = _stage_model()
    model(synthetic_batch(), features=synthetic_features())

    cfg = hera_test_config()
    apply_training_stage(model, cfg, epoch=2)
    _, groups = create_optimizer(model, cfg)
    names = {group.name for group in groups}
    audit = parameter_accounting(model)

    assert {"new_hera_modules", "siglip2", "dinov3", "deberta"} <= names
    assert audit["evidence_extractor"]["trainable"] > 0
    assert audit["router"]["trainable"] > 0
    assert audit["hgt"]["trainable"] > 0
    assert audit["hypothesis_embeddings"]["trainable"] > 0
    assert audit["attribution_head"]["trainable"] > 0
    assert audit["image_grounding"]["trainable"] > 0
    assert audit["text_grounding"]["trainable"] > 0


def test_gradient_checkpointing_skips_frozen_stage_a_backbones_and_reenables_stage_b() -> None:
    cfg = hera_test_config()
    cfg["runtime"] = {
        "gradient_checkpointing": {
            "enabled": True,
            "only_trainable_backbones": True,
            "backbones": {"siglip2": True, "dinov3": True, "deberta": True},
        }
    }
    model = HERADGM(
        cfg,
        siglip2=CheckpointableTinyBackbone(),
        dinov3=CheckpointableTinyBackbone(),
        deberta=CheckpointableTinyBackbone(),
        graph_reasoner=FixtureGraphReasoner(hidden_dim=12),
        load_backbones=False,
    )
    model(synthetic_batch(), features=synthetic_features())

    apply_training_stage(model, cfg, epoch=1)
    stage_a = apply_configured_gradient_checkpointing(model, cfg)
    assert stage_a == {"siglip2": False, "dinov3": False, "deberta": False}

    apply_training_stage(model, cfg, epoch=2)
    stage_b = apply_configured_gradient_checkpointing(model, cfg)
    assert stage_b == {"siglip2": True, "dinov3": True, "deberta": True}


def test_hera_nan_guard_fails_with_context() -> None:
    with pytest.raises(FloatingPointError, match="epoch=2, batch=3"):
        _ensure_finite_losses(
            {"total": math.nan},
            epoch=2,
            batch_index=3,
            config=hera_test_config(),
        )
