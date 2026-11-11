"""Checkpoint save/load and retention for baselines."""

from __future__ import annotations

import random
import os
import json
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from heri.training.state import TrainingState
from heri.training.optimizer import ensure_no_uninitialized_parameters
from heri.evaluation.protocol import EVALUATION_PROTOCOL


def validate_baseline_checkpoint_config(
    current_config: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject legacy/cross-variant baseline checkpoints before loading weights."""

    current_type = current_config.get("model", {}).get("baseline_type")
    for section, key in (("dataset", "name"), ("dataset", "root")):
        current_value = current_config.get(section, {}).get(key)
        checkpoint_value = checkpoint_config.get(section, {}).get(key)
        if current_value != checkpoint_value:
            raise ValueError(
                f"{context} baseline {section}.{key} mismatch: "
                f"{current_value!r} != {checkpoint_value!r}"
            )
    current_hypotheses = (
        current_config.get("model", {}).get("hypotheses", {}).get("order", ["FS", "FA", "TS", "TA"])
    )
    checkpoint_hypotheses = (
        checkpoint_config.get("model", {})
        .get("hypotheses", {})
        .get("order", ["FS", "FA", "TS", "TA"])
    )
    if list(current_hypotheses) != list(checkpoint_hypotheses):
        raise ValueError(f"{context} baseline hypothesis ontology mismatch")
    if current_config.get("dataset_hashes", {}) != checkpoint_config.get("dataset_hashes", {}):
        raise ValueError(f"{context} baseline dataset hash mismatch")
    if current_config.get("scientific_study", {}) != checkpoint_config.get("scientific_study", {}):
        raise ValueError(f"{context} checkpoint scientific-study mismatch")
    for key in ("binary_aggregation",):
        current = current_config.get("model", {}).get(key, "independent_head")
        previous = checkpoint_config.get("model", {}).get(key, "independent_head")
        if current != previous:
            raise ValueError(f"{context} baseline {key} mismatch: {current} != {previous}")
    checkpoint_type = checkpoint_config.get("model", {}).get("baseline_type")
    if checkpoint_type != current_type:
        raise ValueError(
            f"{context} baseline checkpoint mismatch: current={current_type!r}, "
            f"checkpoint={checkpoint_type!r}"
        )
    current_version = current_config.get("project", {}).get("architecture_version")
    checkpoint_version = checkpoint_config.get("project", {}).get("architecture_version")
    if current_version != "heri_baseline" or checkpoint_version != current_version:
        raise ValueError(
            f"{context} baseline architecture mismatch: current={current_version!r}, "
            f"checkpoint={checkpoint_version!r}; final baselines require "
            "'heri_baseline'"
        )
    if context.lower().startswith("resume"):
        validate_resume_identity(current_config, checkpoint_config)


def validate_resume_identity(current: Mapping[str, Any], previous: Mapping[str, Any]) -> None:
    """A training continuation must retain both model and data-order seeds."""

    for section, key in (("project", "seed"), ("dataloader", "seed")):
        if current.get(section, {}).get(key) != previous.get(section, {}).get(key):
            raise ValueError(f"Resume {section}.{key} mismatch")


def dataloader_states(dataloaders: Mapping[str, Any] | None) -> dict[str, Any]:
    """Capture loader-local random streams, independent of global torch RNG."""

    states: dict[str, Any] = {}
    for name, loader in (dataloaders or {}).items():
        generator = getattr(loader, "generator", None)
        sampler = getattr(loader, "sampler", None)
        states[name] = {
            "generator": generator.get_state() if generator is not None else None,
            "sampler_generator": (
                sampler.generator.get_state()
                if getattr(sampler, "generator", None) is not None
                else None
            ),
            "sampler": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
        }
    return states


def restore_dataloader_states(states: Mapping[str, Any], dataloaders: Mapping[str, Any]) -> None:
    """Restore epoch-boundary data order before constructing the next iterator."""

    for name, state in states.items():
        if name not in dataloaders:
            continue
        loader = dataloaders[name]
        if state.get("generator") is not None:
            loader.generator.set_state(state["generator"])
        if state.get("sampler_generator") is not None:
            loader.sampler.generator.set_state(state["sampler_generator"])
        if state.get("sampler") is not None:
            sampler = loader.sampler
            if not hasattr(sampler, "load_state_dict"):
                raise ValueError(f"Cannot restore sampler state for {name}")
            sampler.load_state_dict(state["sampler"])


def rng_state() -> dict[str, Any]:
    """Capture RNG state where practical."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state where available."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    state: TrainingState,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    precision: str,
    scheduler_plan: Mapping[str, Any] | None = None,
    dataloaders: Mapping[str, Any] | None = None,
) -> str:
    """Save a complete resume-capable checkpoint and return its unique identity."""

    ensure_no_uninitialized_parameters(model, context="checkpoint save")
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_id = str(uuid.uuid4())
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "checkpoint_id": checkpoint_id,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_metric": state.best_metric,
        "best_score": list(state.best_score) if state.best_score is not None else None,
        "best_checkpoint": str(state.best_checkpoint)
        if state.best_checkpoint is not None
        else None,
        "early_stopping_best_metric": state.early_stopping_best_metric,
        "early_stopping_bad_epochs": state.early_stopping_bad_epochs,
        "thresholds": state.thresholds.as_dict(),
        "metrics": dict(metrics),
        "config": dict(config),
        "precision": precision,
        "scheduler_plan": dict(scheduler_plan) if scheduler_plan is not None else None,
        "rng_state": rng_state(),
        "dataloader_states": dataloader_states(dataloaders),
    }
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    # The metadata binds a concrete checkpoint without unpickling it during audits.
    from heri.experiments.provenance import canonical_config_sha256, checkpoint_metadata_path
    from heri.utils.hashing import sha256_file

    environment_path = path.parent.parent / "environment.json"
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.is_file()
        else {}
    )
    metadata = {
        "schema": "heri_checkpoint_metadata",
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": sha256_file(path, progress=False),
        "effective_config_sha256": canonical_config_sha256(config),
        "effective_config": dict(config),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "alignment_objective": config.get("loss", {})
        .get("alignment", {})
        .get("aggregation", "not_applicable"),
        "source_sha256": environment.get("source_sha256"),
        "git_commit": environment.get("git_commit"),
        "epoch": state.epoch,
        "global_step": state.global_step,
    }
    metadata_path = checkpoint_metadata_path(path)
    metadata_temporary = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        metadata_temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(metadata_temporary, metadata_path)
    finally:
        metadata_temporary.unlink(missing_ok=True)
    return checkpoint_id


def load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load checkpoint and restore all requested training state."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return restore_checkpoint(
        checkpoint=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        restore_rng=restore_rng,
    )


def restore_checkpoint(
    *,
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore an already-loaded checkpoint without a second disk read."""

    if (restore_rng or optimizer is not None or scheduler is not None) and checkpoint.get(
        "evaluation_protocol"
    ) != EVALUATION_PROTOCOL:
        raise ValueError(
            "Cannot resume training with legacy metric-selection state. "
            "Use weights-only evaluation (restore_rng=False) or start an isolated current-protocol run."
        )
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng and "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint
