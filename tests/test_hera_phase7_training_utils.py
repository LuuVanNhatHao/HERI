from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from hera_phase7_fixtures import FixtureGraphReasoner, TinyBackbone, hera_test_config
from phase2_fixtures import synthetic_batch, synthetic_features
from src.models.hera import HERADGM
from src.training.hera_trainer import (
    _cuda_memory_postfix,
    _cuda_memory_summary,
    _early_stopping_improved,
    _ensure_finite_losses,
    _resolve_hera_resume_checkpoint,
    apply_configured_gradient_checkpointing,
)
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


def test_cuda_memory_helpers_are_inactive_for_cpu_device() -> None:
    assert _cuda_memory_summary(torch.device("cpu")) is None
    assert _cuda_memory_postfix(torch.device("cpu")) is None


def test_hera_auto_resume_selects_latest_last_checkpoint(tmp_path: Path) -> None:
    older = tmp_path / "hera_seed42_old" / "checkpoints" / "last.pt"
    newer = tmp_path / "hera_seed42_new" / "checkpoints" / "last.pt"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    cfg = hera_test_config()
    cfg["output"] = {"root": str(tmp_path)}
    cfg["training"]["resume_from"] = None
    cfg["training"]["auto_resume"] = {
        "enabled": True,
        "run_dir": None,
        "checkpoint_name": "last.pt",
    }

    assert _resolve_hera_resume_checkpoint(cfg) == newer


def test_hera_explicit_resume_overrides_auto_resume(tmp_path: Path) -> None:
    explicit = tmp_path / "manual.pt"
    automatic = tmp_path / "hera_seed42_new" / "checkpoints" / "last.pt"
    automatic.parent.mkdir(parents=True)
    explicit.write_bytes(b"manual")
    automatic.write_bytes(b"auto")
    cfg = hera_test_config()
    cfg["output"] = {"root": str(tmp_path)}
    cfg["training"]["resume_from"] = str(explicit)
    cfg["training"]["auto_resume"] = {
        "enabled": True,
        "run_dir": None,
        "checkpoint_name": "last.pt",
    }

    assert _resolve_hera_resume_checkpoint(cfg) == explicit


def test_hera_early_stopping_improvement_respects_mode_and_delta() -> None:
    assert _early_stopping_improved(current=0.8, best=None, mode="max", min_delta=0.0)
    assert _early_stopping_improved(current=0.81, best=0.8, mode="max", min_delta=0.005)
    assert not _early_stopping_improved(current=0.804, best=0.8, mode="max", min_delta=0.005)
    assert _early_stopping_improved(current=0.79, best=0.8, mode="min", min_delta=0.005)
    assert not _early_stopping_improved(current=0.798, best=0.8, mode="min", min_delta=0.005)


def test_hera_early_stopping_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported early stopping mode"):
        _early_stopping_improved(current=0.8, best=0.7, mode="sideways", min_delta=0.0)
