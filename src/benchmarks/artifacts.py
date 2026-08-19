"""Run discovery, fingerprints, and crash-safe benchmark checkpoint I/O."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from src.training.checkpoint import rng_state

from .contracts import SignalContract


def create_benchmark_run_dir(
    config: Mapping[str, Any],
    *,
    resume_checkpoint: Path | None,
) -> Path:
    if resume_checkpoint is not None and resume_checkpoint.parent.name == "checkpoints":
        run_dir = resume_checkpoint.parent.parent
        _ensure_subdirs(run_dir)
        return run_dir
    root = Path(str(config["output"]["root"]))
    adapter = str(config["dataset"]["adapter"]).casefold()
    seed = int(config["project"]["seed"])
    run_id = str(
        config["project"].get("run_id")
        or f"hera_{adapter}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Benchmark run directory already exists: {run_dir}")
    _ensure_subdirs(run_dir)
    return run_dir


def resolve_resume_checkpoint(config: Mapping[str, Any]) -> Path | None:
    explicit = config["training"].get("resume_from")
    if explicit not in (None, ""):
        path = Path(str(explicit))
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured resume checkpoint does not exist: {path}"
            )
        return path
    auto = config["training"]["auto_resume"]
    if not bool(auto["enabled"]):
        return None
    checkpoint_name = str(auto["checkpoint_name"])
    run_dir = auto.get("run_dir")
    if run_dir not in (None, ""):
        path = Path(str(run_dir)) / "checkpoints" / checkpoint_name
        if not path.is_file():
            raise FileNotFoundError(
                f"Auto-resume run has no {checkpoint_name}: {run_dir}"
            )
        return path
    root = Path(str(config["output"]["root"]))
    candidates = [
        path for path in root.glob(f"*/checkpoints/{checkpoint_name}") if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def resolve_evaluation_checkpoint(config: Mapping[str, Any]) -> Path:
    evaluation = config["evaluation"]
    explicit = evaluation.get("checkpoint_path")
    if explicit not in (None, ""):
        path = Path(str(explicit))
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation checkpoint does not exist: {path}")
        return path
    run_dir = evaluation.get("run_dir")
    if run_dir not in (None, ""):
        path = Path(str(run_dir)) / "checkpoints" / "best_model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation run has no best_model.pt: {run_dir}")
        return path
    root = Path(str(config["output"]["root"]))
    candidates = [
        path for path in root.glob("*/checkpoints/best_model.pt") if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No benchmark best_model.pt found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def benchmark_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    manifest = dataset.get("split_manifest", {}).get("path")
    manifest_path = Path(str(manifest)) if manifest not in (None, "") else None
    return {
        "task_type": config["task"]["type"],
        "signals": SignalContract.from_config(config).as_dict(),
        "dataset_adapter": dataset["adapter"],
        "dataset_name": dataset["name"],
        "split_protocol": dataset["split_protocol"],
        "cross_split_duplicate_policy": dataset["cross_split_duplicate_policy"],
        "primary_target": config["task"]["primary_target"],
        "text_fields": list(dataset["text"]["fields"]),
        "tokenizer_name": config["tokenizer"]["name"],
        "tokenizer_max_length": int(config["tokenizer"]["max_length"]),
        "image_size": int(config["image"]["size"]),
        "backbones": {
            name: dict(values) for name, values in config["backbones"].items()
        },
        "architecture": {
            "evidence": dict(config["evidence"]),
            "routing": dict(config["routing"]),
            "graph": dict(config["graph"]),
            "model": dict(config["model"]),
        },
        "objective": dict(config["loss"]),
        "optimization_contract": {
            "micro_batch_size": int(config["training"]["micro_batch_size"]),
            "gradient_accumulation_steps": int(
                config["training"]["gradient_accumulation_steps"]
            ),
            "max_grad_norm": config["training"].get("max_grad_norm"),
            "stages": dict(config["training"]["stages"]),
            "optimizer": dict(config["optimizer"]),
            "scheduler": dict(config["scheduler"]),
            "precision": config["runtime"]["precision"],
            "model_dtype": config["runtime"]["model_dtype"],
            "gradient_checkpointing": dict(config["runtime"]["gradient_checkpointing"]),
        },
        "manifest_sha256": (
            sha256_file(manifest_path)
            if manifest_path is not None and manifest_path.is_file()
            else None
        ),
    }


def assert_checkpoint_signature(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    expected = benchmark_signature(config)
    actual = checkpoint.get("benchmark_signature")
    if actual != expected:
        raise ValueError(
            "Benchmark checkpoint/config contract mismatch. "
            f"checkpoint={actual!r}, current={expected!r}"
        )


def save_last_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    state: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    precision: str,
    scheduler_plan: Mapping[str, Any] | None,
) -> None:
    payload = {
        "checkpoint_type": "resume",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        **dict(state),
        "metrics": dict(metrics),
        "config": dict(config),
        "precision": precision,
        "scheduler_plan": dict(scheduler_plan) if scheduler_plan is not None else None,
        "benchmark_signature": benchmark_signature(config),
        "rng_state": rng_state(),
    }
    _save_payload(payload, path=path, config=config)


def save_best_model(
    *,
    path: Path,
    model: torch.nn.Module,
    state: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    precision: str,
) -> None:
    payload = {
        "checkpoint_type": "evaluation_weights",
        "model": model.state_dict(),
        **dict(state),
        "metrics": dict(metrics),
        "config": dict(config),
        "precision": precision,
        "benchmark_signature": benchmark_signature(config),
    }
    _save_payload(payload, path=path, config=config)


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_payload(
    payload: Mapping[str, Any], *, path: Path, config: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    minimum_free = float(config["checkpointing"].get("minimum_free_gib", 0.0))
    free_gib = shutil.disk_usage(path.parent).free / 1024**3
    estimated_payload_gib = _tensor_payload_bytes(payload) * 1.05 / 1024**3
    required_free_gib = max(minimum_free, estimated_payload_gib + 1.0)
    if free_gib < required_free_gib:
        raise OSError(
            f"Refusing checkpoint write with insufficient free disk: "
            f"free={free_gib:.2f} GiB, required={required_free_gib:.2f} GiB, "
            f"estimated_payload={estimated_payload_gib:.2f} GiB, path={path}"
        )
    atomic = bool(config["checkpointing"].get("atomic_write", True))
    target = path.with_suffix(path.suffix + ".tmp") if atomic else path
    try:
        torch.save(dict(payload), target)
        if atomic:
            target.replace(path)
    except Exception:
        if atomic and target.exists():
            target.unlink()
        raise


def _tensor_payload_bytes(value: Any) -> int:
    """Conservatively estimate serialized tensor bytes before an atomic write."""

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_payload_bytes(item) for item in value)
    return 0


def _ensure_subdirs(run_dir: Path) -> None:
    for name in ("checkpoints", "metrics", "thresholds", "predictions", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
