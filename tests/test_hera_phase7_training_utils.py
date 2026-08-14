from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hera_phase7_fixtures import FixtureGraphReasoner, TinyBackbone, hera_test_config
from phase2_fixtures import synthetic_batch, synthetic_features
from src.models.hera import HERADGM
from src.training.hera_trainer import (
    HERATrainer,
    _cuda_memory_postfix,
    _cuda_memory_summary,
    _early_stopping_improved,
    _ensure_finite_losses,
    _recover_early_stopping_state,
    _resolve_hera_resume_checkpoint,
    apply_configured_gradient_checkpointing,
)
from src.training.optimizer import (
    apply_training_stage,
    create_optimizer,
    migrate_optimizer_state,
    parameter_accounting,
)
from src.training.state import TrainingState


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


def test_stage_b_optimizer_rebuild_preserves_stage_a_adam_state() -> None:
    model = _stage_model()
    model(synthetic_batch(), features=synthetic_features())
    cfg = hera_test_config()
    apply_training_stage(model, cfg, epoch=1)
    stage_a, _ = create_optimizer(model, cfg)
    for group in stage_a.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    stage_a.step()
    common_parameter = stage_a.param_groups[0]["params"][0]
    expected_exp_avg = stage_a.state[common_parameter]["exp_avg"].clone()

    apply_training_stage(model, cfg, epoch=2)
    stage_b, _ = create_optimizer(model, cfg)
    migrated = migrate_optimizer_state(stage_a, stage_b)

    assert migrated > 0
    assert common_parameter in stage_b.state
    assert torch.equal(stage_b.state[common_parameter]["exp_avg"], expected_exp_avg)
    assert {group["name"] for group in stage_b.param_groups} >= {
        "new_hera_modules", "siglip2", "dinov3", "deberta"
    }


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


def test_early_stopping_waits_for_stage_b_grace_epochs() -> None:
    trainer = HERATrainer.__new__(HERATrainer)
    trainer.config = {
        "training": {
            "stages": {"stage_a_freeze_epochs": 1},
            "early_stopping": {
                "enabled": True,
                "metric": "cf1",
                "mode": "max",
                "patience": 1,
                "min_delta": 0.001,
                "minimum_stage_b_epochs": 2,
            },
        }
    }
    trainer.state = TrainingState()
    trainer.state.epoch = 1
    assert not trainer._update_early_stopping({"cf1": 0.8})
    trainer.state.epoch = 2
    assert not trainer._update_early_stopping({"cf1": 0.7})
    trainer.state.epoch = 3
    assert not trainer._update_early_stopping({"cf1": 0.6})
    assert trainer.state.early_stopping_bad_epochs == 0
    trainer.state.epoch = 4
    assert trainer._update_early_stopping({"cf1": 0.5})


def test_legacy_early_stopping_state_is_recovered_from_history(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    history.write_text(
        "epoch,val_cf1\n"
        "6,0.7855\n"
        "7,0.7900\n"
        "8,0.7950\n"
        "9,0.7948\n"
        "10,0.7940\n",
        encoding="utf-8",
    )

    best, bad_epochs = _recover_early_stopping_state(
        history,
        {"metric": "cf1", "mode": "max", "min_delta": 0.001},
    )

    assert best == pytest.approx(0.7950)
    assert bad_epochs == 2


def test_legacy_epoch_10_checkpoint_extends_to_20_without_lr_jump() -> None:
    trainer = HERATrainer.__new__(HERATrainer)
    trainer.config = {
        "scheduler": {
            "type": "linear_warmup",
            "warmup_steps": 500,
            "resume_horizon_policy": "continue_from_checkpoint_lr",
        }
    }
    trainer.model = torch.nn.Linear(2, 1)
    trainer.precision = SimpleNamespace(scaler=None)
    trainer._scheduler_plan = None
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    checkpoint_optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    checkpoint_optimizer.param_groups[0]["lr"] = 3e-5
    checkpoint_optimizer.param_groups[0]["initial_lr"] = 1e-4
    checkpoint = {
        "model": trainer.model.state_dict(),
        "optimizer": checkpoint_optimizer.state_dict(),
        "scheduler": None,
        "scaler": None,
        "epoch": 10,
        "global_step": 65_060,
        "config": {"training": {"epochs": 10}},
    }

    scheduler = trainer._restore_resume_checkpoint(
        checkpoint=checkpoint,
        optimizer=optimizer,
        total_epochs=20,
        total_steps=130_120,
        optimizer_steps_per_epoch=6_506,
        resume_checkpoint_epoch=10,
        crosses_stage_boundary=False,
    )

    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert trainer._scheduler_plan["type"] == "linear_decay_continuation"
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 3e-5
