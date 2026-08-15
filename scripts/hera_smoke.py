"""Run a real Phase-1 to Phase-7 HERA smoke check from YAML."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import create_dataloader
from src.evaluation.evaluator import move_batch_to_device
from src.losses.hera_components import HERAMultitaskLoss
from src.models.heads.noisy_or import noisy_or_from_probabilities
from src.models.hera import HERADGM
from src.training.hera_trainer import apply_configured_gradient_checkpointing
from src.training.optimizer import apply_training_stage, create_optimizer, ensure_no_uninitialized_parameters
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_hera_config
from src.utils.environment import collect_environment_metadata
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/hera_smoke.py <config.yaml>", file=sys.stderr)
        return 2
    config = load_yaml_config(sys.argv[1])
    validate_hera_config(config, mode="smoke")
    setup_logging(str(config["logging"]["level"]))
    set_seed(int(config["project"]["seed"]))

    output_root = Path(str(config["output"]["root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_root / "effective_config.yaml")
    device = torch.device(str(config["runtime"]["device"]))
    precision = PrecisionManager(precision=str(config["runtime"]["precision"]), device=device)
    model = HERADGM.from_config(config, load_backbones=True).to(device)
    gradient_checkpointing = apply_configured_gradient_checkpointing(model, config)
    apply_training_stage(model, config, epoch=1)
    loss_fn = HERAMultitaskLoss(config["loss"])
    if model.deberta is None or not hasattr(model.deberta, "tokenizer"):
        raise RuntimeError("HERA smoke requires an attached DeBERTa fast tokenizer")
    loader = create_dataloader(
        config,
        split=str(config["smoke"]["split"]),
        tokenizer=model.deberta.tokenizer,
        max_samples=int(config["smoke"]["max_samples"]),
        samples_per_class=config["smoke"].get("samples_per_class"),
    )
    backward = bool(config["smoke"].get("differentiability", True))
    model.train(backward)
    optimizer = None
    inspect_batches = int(config["smoke"]["inspect_batches"])
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    iterator = tqdm(
        loader,
        total=min(inspect_batches, len(loader)),
        desc="HERA smoke",
        disable=not bool(config["progress"]["enabled"]),
    )
    for batch_index, raw_batch in enumerate(iterator, start=1):
        if batch_index > inspect_batches:
            break
        batch = move_batch_to_device(raw_batch, device)
        with torch.set_grad_enabled(backward):
            with precision.autocast():
                output = model(batch)
                losses = loss_fn(output, batch)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("Non-finite Phase-7 HERA smoke loss")
        if backward:
            if optimizer is None:
                ensure_no_uninitialized_parameters(model, context="Phase-7 HERA smoke optimizer creation")
                optimizer, _ = create_optimizer(model, config)
            optimizer.zero_grad(set_to_none=True)
            precision.backward(losses.total)
            _assert_gradient_connectivity(model)
            precision.optimizer_step(optimizer)
        record = _diagnostics(batch_index, output, losses)
        records.append(record)
        print(json.dumps(record, indent=2))

    ensure_no_uninitialized_parameters(model, context="Phase-7 HERA smoke completion")
    report = {
        "device": str(device),
        "precision_requested": precision.requested,
        "precision_actual": precision.actual,
        "runtime_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "routing_enabled": bool(config["routing"]["enabled"]),
        "gradient_checkpointing": gradient_checkpointing,
        "binary_aggregation": "noisy_or",
        "binary_shortcut_count": model.count_independent_binary_shortcuts(),
        "benchmark_performance_claim": False,
        "batches": records,
    }
    (output_root / "hera_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / "environment.json").write_text(
        json.dumps(collect_environment_metadata(ROOT), indent=2),
        encoding="utf-8",
    )
    return 0


def _diagnostics(batch_index: int, output: Any, losses: Any) -> dict[str, Any]:
    reference_fake = noisy_or_from_probabilities(output.atomic_probabilities.float())
    noisy_or_error = torch.max(torch.abs(reference_fake - output.binary_probability.float()))
    if not torch.isfinite(noisy_or_error):
        raise FloatingPointError("Non-finite Noisy-OR consistency error")
    return {
        "batch_index": batch_index,
        "hypothesis_shape": list(output.hypothesis_features.shape),
        "atomic_logits_shape": list(output.atomic_logits.shape),
        "binary_probability_shape": list(output.binary_probability.shape),
        "image_grounding_shape": list(output.image_grounding_logits.shape),
        "text_grounding_shape": list(output.text_grounding_logits.shape),
        "noisy_or_max_abs_error": float(noisy_or_error.detach().cpu()),
        "router_diagnostics": output.metadata.get("router_batch_summary"),
        "losses": losses.as_dict(),
    }


def _assert_gradient_connectivity(model: HERADGM) -> None:
    trainable_groups = {
        "router": model.router,
        "graph": model.graph_reasoner,
        "decision": model.decision,
    }
    for name, module in trainable_groups.items():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError(f"No gradients reached HERA smoke module: {name}")
        if not all(torch.isfinite(grad).all() for grad in gradients):
            raise FloatingPointError(f"Non-finite gradients in HERA smoke module: {name}")


if __name__ == "__main__":
    raise SystemExit(main())
