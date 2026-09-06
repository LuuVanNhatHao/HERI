"""Run a real-model Phase-2 smoke check from YAML."""

from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import create_dataloader
from src.evaluation.evaluator import move_batch_to_device
from src.losses.baseline_losses import BaselineMultitaskLoss
from src.models.baselines import build_baseline_model
from src.training.optimizer import apply_training_stage, create_optimizer, parameter_accounting
from src.training.precision import PrecisionManager
from src.utils.config import load_yaml_config, validate_baseline_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def _shape(value: Any) -> str:
    if hasattr(value, "shape"):
        return str(tuple(value.shape))
    return type(value).__name__


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/model_smoke.py <config.yaml>", file=sys.stderr)
        return 2
    config = load_yaml_config(sys.argv[1])
    validate_baseline_config(config, mode="smoke")
    setup_logging(str(config["logging"]["level"]))
    set_seed(int(config["project"]["seed"]))
    device = torch.device(str(config["runtime"]["device"]))
    model = build_baseline_model(config, load_backbones=True).to(device)
    apply_training_stage(model, config, epoch=1)
    tokenizer = model.deberta.tokenizer
    loader = create_dataloader(
        config,
        split=str(config["smoke"]["split"]),
        tokenizer=tokenizer,
        max_samples=int(config["smoke"]["max_samples"]),
        samples_per_class=int(config["smoke"]["samples_per_class"]),
    )
    loss_fn = BaselineMultitaskLoss(config["loss"]["weights"])
    precision = PrecisionManager(precision=str(config["runtime"]["precision"]), device=device)
    backward = bool(config["smoke"].get("backward", True))
    inspect_batches = int(config["smoke"]["inspect_batches"])
    model.train(backward)
    optimizer = None
    total = min(inspect_batches, len(loader))
    iterator = tqdm(islice(loader, total), total=total, desc="Model smoke")
    for batch_index, raw_batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(raw_batch, device)
        with precision.autocast():
            output = model(batch)
            losses = loss_fn(output, batch)
        if optimizer is None:
            optimizer, group_summaries = create_optimizer(model, config)
            print(f"parameter_accounting: {parameter_accounting(model)}")
            print(f"optimizer_groups: {[summary.__dict__ for summary in group_summaries]}")
        if backward:
            optimizer.zero_grad(set_to_none=True)
            precision.backward(losses.total)
            precision.optimizer_step(optimizer)
        print(f"batch_index: {batch_index}")
        print(f"device: {device}")
        print(f"dtype: {output.binary_logit.dtype}")
        print(f"binary_logit: {_shape(output.binary_logit)}")
        print(f"atomic_logits: {_shape(output.atomic_logits)}")
        print(f"image_grounding_logits: {_shape(output.image_grounding_logits)}")
        print(f"text_grounding_logits: {_shape(output.text_grounding_logits)}")
        print(f"loss_total: {float(losses.total.detach().cpu()):.6f}")
        if torch.cuda.is_available():
            print(f"peak_vram_bytes: {torch.cuda.max_memory_allocated()}")
        if not torch.isfinite(losses.total):
            raise FloatingPointError("Non-finite model smoke loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
