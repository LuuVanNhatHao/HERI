"""No-checkpoint full-fine-tune capacity probe for binary HERA benchmarks."""

from __future__ import annotations

import copy
import time
from typing import Any, Mapping

import torch
from tqdm.auto import tqdm

from src.evaluation.evaluator import move_batch_to_device
from src.training.hera_trainer import apply_configured_gradient_checkpointing
from src.training.optimizer import apply_training_stage, create_optimizer
from src.training.precision import PrecisionManager

from .loader import create_binary_benchmark_dataloader
from .loss import BinaryBenchmarkLoss
from .model import BinaryBenchmarkHERA
from .training_utils import full_ft_gradient_coverage


BYTES_PER_GIB = 1024**3


def build_binary_capacity_config(
    config: Mapping[str, Any],
    *,
    batch_size: int,
    batches: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    if min(batch_size, batches, gradient_accumulation_steps) <= 0:
        raise ValueError(
            "batch_size, batches, and gradient_accumulation_steps must be positive"
        )
    if batches % gradient_accumulation_steps:
        raise ValueError("batches must be divisible by gradient_accumulation_steps")
    capacity = copy.deepcopy(dict(config))
    capacity["training"]["epochs"] = 1
    capacity["training"]["micro_batch_size"] = int(batch_size)
    capacity["training"]["gradient_accumulation_steps"] = int(
        gradient_accumulation_steps
    )
    capacity["training"]["resume_from"] = None
    capacity["training"]["auto_resume"]["enabled"] = False
    capacity["training"]["early_stopping"]["enabled"] = False
    capacity["training"]["stages"] = {
        "stage_a_freeze_epochs": 0,
        "stage_b_full_unfreeze": {"enabled": True, "start_epoch": 1},
    }
    capacity["dataloader"]["batch_size"] = int(batch_size)
    capacity["dataloader"]["shuffle"]["train"] = False
    capacity["runtime"]["max_train_batches"] = int(batches)
    return capacity


def run_binary_capacity_test(
    config: Mapping[str, Any],
    *,
    batches: int,
) -> dict[str, Any]:
    device = _indexed_cuda_device(str(config["runtime"]["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Binary benchmark capacity testing requires an available CUDA GPU"
        )
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    precision = PrecisionManager(
        precision=str(config["runtime"]["precision"]), device=device
    )
    model = BinaryBenchmarkHERA.from_config(config, load_backbones=True).to(device)
    apply_training_stage(model, config, epoch=1)
    checkpointing = apply_configured_gradient_checkpointing(model, config)
    if model.deberta is None or not hasattr(model.deberta, "tokenizer"):
        raise RuntimeError("Capacity test requires the configured fast tokenizer")
    batch_size = int(config["dataloader"]["batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    loader = create_binary_benchmark_dataloader(
        config,
        split="train",
        tokenizer=model.deberta.tokenizer,
        max_samples=batch_size * int(batches),
    )
    loss_fn = BinaryBenchmarkLoss(config)
    model.train()
    optimizer: torch.optim.Optimizer | None = None
    optimizer_groups: list[dict[str, Any]] = []
    completed_batches = 0
    completed_samples = 0
    completed_steps = 0
    snapshots = [_memory_snapshot(device, stage="model_loaded")]
    coverage: dict[str, Any] | None = None
    started = time.perf_counter()
    for batch_index, raw_batch in enumerate(
        tqdm(loader, total=batches, desc="Binary benchmark full-FT capacity"),
        start=1,
    ):
        if batch_index > batches:
            break
        capacity_text_tokens = _capacity_text_tokens(config)
        if capacity_text_tokens is not None:
            raw_batch = expand_capacity_text_batch(
                raw_batch,
                target_length=capacity_text_tokens,
            )
        batch = move_batch_to_device(raw_batch, device)
        with precision.autocast():
            output = model(batch)
            losses = loss_fn(output, batch)
        if optimizer is None:
            optimizer, summaries = create_optimizer(model, config)
            optimizer_groups = [summary.__dict__ for summary in summaries]
            optimizer.zero_grad(set_to_none=True)
        precision.backward(losses.total / accumulation)
        coverage = full_ft_gradient_coverage(
            model,
            minimum_element_coverage=float(
                config["runtime"]["gradient_audit"]["minimum_element_coverage"]
            ),
        )
        should_step = batch_index % accumulation == 0
        if should_step:
            maximum = config["training"].get("max_grad_norm")
            if maximum is not None:
                if precision.actual == "fp16":
                    precision.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(maximum), error_if_nonfinite=True
                )
            precision.optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
        completed_batches += 1
        completed_samples += int(batch["image"].shape[0])
        snapshot = _memory_snapshot(device, stage=f"batch_{batch_index}_backward")
        snapshot["loss"] = float(losses.total.detach().cpu())
        snapshots.append(snapshot)
    duration = time.perf_counter() - started
    if completed_batches != batches or completed_steps != batches // accumulation:
        raise RuntimeError(
            f"Capacity test incomplete: batches={completed_batches}/{batches}, "
            f"optimizer_steps={completed_steps}/{batches // accumulation}"
        )
    return {
        "status": "pass",
        "dataset": config["dataset"]["name"],
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "completed_batches": completed_batches,
        "completed_optimizer_steps": completed_steps,
        "completed_samples": completed_samples,
        "capacity_text_tokens": _capacity_text_tokens(config),
        "duration_seconds": duration,
        "samples_per_second": completed_samples / max(duration, 1e-9),
        "precision": precision.actual,
        "gradient_checkpointing": checkpointing,
        "optimizer_groups": optimizer_groups,
        "backbone_gradient_coverage": coverage,
        "snapshots": snapshots,
        "checkpoint_written": False,
    }


def expand_capacity_text_batch(
    batch: Mapping[str, Any],
    *,
    target_length: int,
) -> dict[str, Any]:
    """Expand structural text tensors to a worst-case active sequence length.

    This is capacity-test-only data. Repeating the final valid token makes the
    attention mask fully active, preventing unpadding kernels from understating
    long-context activation memory.
    """

    expanded = dict(batch)
    input_ids = batch["text_input_ids"]
    attention_mask = batch["text_attention_mask"].to(torch.bool)
    word_ids = batch["text_word_ids"]
    grounding_mask = batch["text_grounding_mask"].to(torch.bool)
    if input_ids.ndim != 2 or target_length <= 0:
        raise ValueError("Capacity text expansion requires [B,L] IDs and a positive target")
    if input_ids.shape[1] >= target_length:
        return expanded
    rows = int(input_ids.shape[0])
    new_ids = input_ids.new_empty((rows, target_length))
    new_attention = torch.ones(
        (rows, target_length), dtype=torch.bool, device=attention_mask.device
    )
    new_word_ids = word_ids.new_empty((rows, target_length))
    new_grounding = torch.ones(
        (rows, target_length), dtype=torch.bool, device=grounding_mask.device
    )
    current_length = int(input_ids.shape[1])
    new_ids[:, :current_length] = input_ids
    new_attention[:, :current_length] = attention_mask
    new_word_ids[:, :current_length] = word_ids
    new_grounding[:, :current_length] = grounding_mask
    for row in range(rows):
        valid_positions = attention_mask[row].nonzero(as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            raise ValueError("Capacity text expansion encountered an empty sequence")
        final_position = int(valid_positions[-1])
        padding_positions = (~attention_mask[row]).nonzero(as_tuple=False).flatten()
        appended_positions = torch.arange(
            current_length,
            target_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        fill_positions = torch.cat((padding_positions, appended_positions))
        new_ids[row, fill_positions] = input_ids[row, final_position]
        new_attention[row, fill_positions] = True
        new_grounding[row, fill_positions] = True
        valid_word_ids = word_ids[row][word_ids[row] >= 0]
        next_word_id = int(valid_word_ids.max()) + 1 if valid_word_ids.numel() else 0
        new_word_ids[row, fill_positions] = torch.arange(
            next_word_id,
            next_word_id + fill_positions.numel(),
            dtype=word_ids.dtype,
            device=word_ids.device,
        )
    expanded["text_input_ids"] = new_ids
    expanded["text_attention_mask"] = new_attention
    expanded["text_word_ids"] = new_word_ids
    expanded["text_grounding_mask"] = new_grounding
    return expanded


def _capacity_text_tokens(config: Mapping[str, Any]) -> int | None:
    if str(config["dataset"]["adapter"]).casefold() != "vldbench":
        return None
    long_context = config["dataset"]["text"].get("long_context", {})
    if not bool(long_context.get("enabled", False)):
        return None
    return int(config["tokenizer"]["max_length"])


def _indexed_cuda_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _memory_snapshot(device: torch.device, *, stage: str) -> dict[str, float | str]:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "allocated_gib": torch.cuda.memory_allocated(device) / BYTES_PER_GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / BYTES_PER_GIB,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / BYTES_PER_GIB,
        "free_gib": free / BYTES_PER_GIB,
        "total_gib": total / BYTES_PER_GIB,
    }
