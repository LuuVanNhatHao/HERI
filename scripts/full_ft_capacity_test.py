"""Probe whether canonical HERA full fine-tuning fits in GPU memory."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import create_dataloader
from src.evaluation.evaluator import move_batch_to_device
from src.losses.hera_components import HERAMultitaskLoss
from src.models.hera import HERADGM
from src.training.hera_trainer import apply_configured_gradient_checkpointing
from src.training.optimizer import apply_training_stage, create_optimizer, ensure_no_uninitialized_parameters
from src.training.precision import PrecisionManager
from src.utils.config import load_yaml_config, validate_hera_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


BYTES_PER_GIB = 1024**3


def build_capacity_config(config: Mapping[str, Any], *, batch_size: int, batches: int) -> dict[str, Any]:
    """Return a no-checkpoint config that full-fine-tunes from the first batch."""

    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_size and batches must be positive")
    capacity = copy.deepcopy(dict(config))
    capacity["training"]["epochs"] = 1
    capacity["training"]["micro_batch_size"] = int(batch_size)
    capacity["training"]["gradient_accumulation_steps"] = 1
    capacity["training"]["auto_resume"]["enabled"] = False
    capacity["training"]["resume_from"] = None
    capacity["training"]["early_stopping"]["enabled"] = False
    capacity["training"]["stages"] = {
        "stage_a_freeze_epochs": 0,
        "stage_b_full_unfreeze": {"enabled": True, "start_epoch": 1},
    }
    capacity["dataloader"]["batch_size"] = int(batch_size)
    capacity["dataloader"]["drop_last"] = False
    capacity["dataloader"]["shuffle"]["train"] = False
    capacity["runtime"]["max_train_batches"] = int(batches)
    checkpointing = capacity["runtime"]["gradient_checkpointing"]
    checkpointing["enabled"] = True
    checkpointing["only_trainable_backbones"] = True
    checkpointing["use_reentrant"] = False
    checkpointing["backbones"] = {"siglip2": True, "dinov3": True, "deberta": True}
    capacity["output"]["root"] = "outputs/full_ft_capacity"
    return capacity


def run_capacity_test(config: Mapping[str, Any], *, batches: int) -> dict[str, Any]:
    """Run complete optimization steps without validation or checkpoint writes."""

    device = _resolve_cuda_device(str(config["runtime"]["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full fine-tune capacity testing requires an available CUDA GPU")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    precision = PrecisionManager(precision=str(config["runtime"]["precision"]), device=device)
    model = HERADGM.from_config(config, load_backbones=True).to(device)
    apply_training_stage(model, config, epoch=1)
    _assert_full_backbone_trainability(model)
    gradient_checkpointing = apply_configured_gradient_checkpointing(model, config)
    loss_fn = HERAMultitaskLoss(config["loss"])
    if model.deberta is None or not hasattr(model.deberta, "tokenizer"):
        raise RuntimeError("Capacity test requires the configured DeBERTa tokenizer")
    batch_size = int(config["dataloader"]["batch_size"])
    loader = create_dataloader(
        config,
        split="train",
        tokenizer=model.deberta.tokenizer,
        max_samples=batch_size * int(batches),
    )

    model.train()
    optimizer: torch.optim.Optimizer | None = None
    optimizer_groups: list[dict[str, Any]] = []
    snapshots = [_memory_snapshot(device, stage="model_loaded")]
    completed_batches = 0
    completed_samples = 0
    started = time.perf_counter()
    iterator = tqdm(loader, total=min(batches, len(loader)), desc="Full-FT capacity")
    for batch_index, raw_batch in enumerate(iterator, start=1):
        if batch_index > batches:
            break
        batch = move_batch_to_device(raw_batch, device)
        with precision.autocast():
            output = model(batch)
            losses = loss_fn(output, batch)
        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"Non-finite capacity-test loss at batch {batch_index}")
        if optimizer is None:
            ensure_no_uninitialized_parameters(model, context="full fine-tune capacity optimizer creation")
            optimizer, summaries = create_optimizer(model, config)
            optimizer_groups = [summary.__dict__ for summary in summaries]
        optimizer.zero_grad(set_to_none=True)
        precision.backward(losses.total)
        gradient_coverage = _backbone_gradient_coverage(model)
        snapshots.append(_memory_snapshot(device, stage=f"batch_{batch_index}_backward"))
        max_grad_norm = config["training"].get("max_grad_norm")
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(max_grad_norm),
                error_if_nonfinite=True,
            )
        precision.optimizer_step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        completed_batches += 1
        completed_samples += int(batch["image"].shape[0])
        snapshot = _memory_snapshot(device, stage=f"batch_{batch_index}_optimizer_step")
        snapshot["loss"] = float(losses.total.detach().cpu())
        snapshot["backbone_gradient_coverage"] = gradient_coverage
        snapshots.append(snapshot)
        iterator.set_postfix(peak=f"{snapshot['max_allocated_gib']:.2f} GiB")

    duration = time.perf_counter() - started
    if completed_batches == 0:
        raise RuntimeError("Capacity test dataloader produced no batches")
    return {
        "status": "pass",
        "batch_size": batch_size,
        "completed_batches": completed_batches,
        "completed_samples": completed_samples,
        "duration_seconds": duration,
        "samples_per_second": completed_samples / max(duration, 1e-9),
        "precision": precision.actual,
        "gradient_checkpointing": gradient_checkpointing,
        "optimizer_groups": optimizer_groups,
        "snapshots": snapshots,
        "checkpoint_written": False,
    }


def _resolve_cuda_device(configured_device: str) -> torch.device:
    """Return an indexed CUDA device for APIs that reject bare ``cuda``."""

    device = torch.device(configured_device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _assert_full_backbone_trainability(model: HERADGM) -> None:
    backbones = model.backbone_modules()
    expected = {"siglip2", "dinov3", "deberta"}
    missing = expected.difference(backbones)
    if missing:
        raise RuntimeError(f"Capacity test is missing required backbones: {sorted(missing)}")
    for name, module in backbones.items():
        total = sum(parameter.numel() for parameter in module.parameters())
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        if total == 0 or trainable != total:
            raise RuntimeError(f"Capacity test did not fully unfreeze {name}: {trainable}/{total}")


def _backbone_gradient_coverage(model: HERADGM) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for name, module in model.backbone_modules().items():
        trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
        named_with_gradient = [
            (parameter_name, parameter)
            for parameter_name, parameter in module.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        with_gradient = [parameter for _, parameter in named_with_gradient]
        nonfinite = [
            parameter_name
            for parameter_name, parameter in named_with_gradient
            if not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite:
            raise FloatingPointError(
                f"Non-finite gradients reached full-fine-tune backbone {name}: {nonfinite[:10]}"
            )
        gradient_elements = sum(parameter.grad.numel() for parameter in with_gradient if parameter.grad is not None)
        if gradient_elements == 0:
            raise RuntimeError(f"No gradients reached full-fine-tune backbone {name}")
        coverage[name] = {
            "trainable_tensors": len(trainable),
            "gradient_tensors": len(with_gradient),
            "gradient_elements": gradient_elements,
        }
    return coverage


def _memory_snapshot(device: torch.device, *, stage: str) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "allocated_gib": torch.cuda.memory_allocated(device) / BYTES_PER_GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / BYTES_PER_GIB,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / BYTES_PER_GIB,
        "free_gib": free_bytes / BYTES_PER_GIB,
        "total_gib": total_bytes / BYTES_PER_GIB,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="config/train_hera.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--report", type=Path, default=Path("outputs/full_ft_capacity/report.json"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base = load_yaml_config(args.config)
    config = build_capacity_config(base, batch_size=args.batch_size, batches=args.batches)
    validate_hera_config(config, mode="train")
    setup_logging(str(config["logging"]["level"]))
    set_seed(int(config["project"]["seed"]))
    try:
        report = run_capacity_test(config, batches=args.batches)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        report = {
            "status": "oom",
            "batch_size": args.batch_size,
            "completed_batches": 0,
            "error": str(error),
            "recommended_next_batch_size": max(1, args.batch_size // 2),
            "checkpoint_written": False,
        }
        _write_report(args.report, report)
        print(json.dumps(report, indent=2))
        return 1
    _write_report(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
