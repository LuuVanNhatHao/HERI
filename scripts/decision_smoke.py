"""Run a real Phase-1 to Phase-5 decision smoke check from YAML."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import create_dataloader
from src.evaluation.evaluator import move_batch_to_device
from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor
from src.graph import HypothesisEvidenceGraphReasoner
from src.losses.hera_components import HERAMultitaskLoss
from src.models.baselines.factory import build_backbones
from src.models.hera import HERADecisionModule
from src.models.heads.noisy_or import noisy_or_from_probabilities
from src.models.outputs import BackboneOutputs
from src.training.optimizer import ensure_no_uninitialized_parameters
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_decision_config
from src.utils.environment import collect_environment_metadata
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/decision_smoke.py <config.yaml>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    config = load_yaml_config(config_path)
    validate_decision_config(config, mode="smoke")
    setup_logging(str(config["logging"]["level"]))
    set_seed(int(config["project"]["seed"]))
    output_root = Path(str(config["output"]["root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_root / "effective_config.yaml")

    device = torch.device(str(config["runtime"]["device"]))
    precision = PrecisionManager(precision=str(config["runtime"]["precision"]), device=device)
    backbones = build_backbones(config)
    for module in backbones.values():
        module.to(device)
        module.eval()
    evidence_extractor = HeterogeneousEvidenceExtractor.from_config(config).to(device)
    reasoner = HypothesisEvidenceGraphReasoner.from_config(config).to(device)
    decision = HERADecisionModule.from_config(config).to(device)
    loss_fn = HERAMultitaskLoss(config["loss"])
    tokenizer = backbones["deberta"].tokenizer
    loader = create_dataloader(
        config,
        split=str(config["smoke"]["split"]),
        tokenizer=tokenizer,
        max_samples=int(config["smoke"]["max_samples"]),
        samples_per_class=config["smoke"].get("samples_per_class"),
    )

    backward = bool(config["smoke"].get("differentiability", False))
    trainable_chain = torch.nn.ModuleList([evidence_extractor, reasoner, decision])
    evidence_extractor.train(backward)
    reasoner.train(backward)
    decision.train(backward)
    inspect_batches = int(config["smoke"]["inspect_batches"])
    records: list[dict[str, Any]] = []
    optimizer: torch.optim.Optimizer | None = None
    start = time.perf_counter()
    iterator = tqdm(
        loader,
        total=min(inspect_batches, len(loader)),
        desc="Decision smoke",
        disable=not bool(config["progress"]["enabled"]),
    )
    for batch_index, raw_batch in enumerate(iterator, start=1):
        if batch_index > inspect_batches:
            break
        batch = move_batch_to_device(raw_batch, device)
        with torch.set_grad_enabled(backward):
            with precision.autocast():
                evidence = evidence_extractor(_encode(backbones, batch), EvidenceStructuralMetadata.from_batch(batch))
                graph_output = reasoner(evidence)
                decision_output = decision(graph_output)
                losses = loss_fn(decision_output, batch)
        if backward:
            if optimizer is None:
                ensure_no_uninitialized_parameters(trainable_chain, context="Phase-5 decision smoke optimizer creation")
                optimizer = torch.optim.AdamW(trainable_chain.parameters(), lr=float(config["smoke"]["projection_lr"]))
            optimizer.zero_grad(set_to_none=True)
            precision.backward(losses.total)
            _assert_gradient_connectivity(trainable_chain)
            precision.optimizer_step(optimizer)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("Non-finite Phase-5 decision smoke loss")
        record = _diagnostics(batch_index, decision_output, losses)
        records.append(record)
        print(json.dumps(record, indent=2))

    ensure_no_uninitialized_parameters(trainable_chain, context="Phase-5 decision smoke completion")
    report = {
        "device": str(device),
        "precision_requested": precision.requested,
        "precision_actual": precision.actual,
        "runtime_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "hypothesis_order": list(decision.hypothesis_order),
        "binary_aggregation": "noisy_or",
        "benchmark_performance_claim": False,
        "batches": records,
    }
    (output_root / "decision_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / "environment.json").write_text(json.dumps(collect_environment_metadata(ROOT), indent=2), encoding="utf-8")
    return 0


def _encode(backbones: dict[str, torch.nn.Module], batch: dict[str, Any]) -> BackboneOutputs:
    return BackboneOutputs(
        siglip2=backbones["siglip2"](image=batch["image"], texts=list(batch["text"])),
        dinov3=backbones["dinov3"](image=batch["image"]),
        deberta=backbones["deberta"](
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
        ),
    )


def _diagnostics(batch_index: int, output: Any, losses: Any) -> dict[str, Any]:
    reference_fake = noisy_or_from_probabilities(output.atomic_probabilities.float())
    noisy_or_error = torch.max(torch.abs(reference_fake - output.binary_probability.float()))
    if not torch.isfinite(noisy_or_error):
        raise FloatingPointError("Non-finite Noisy-OR consistency error")
    return {
        "batch_index": batch_index,
        "hypothesis_shape": list(output.hypothesis_features.shape),
        "atomic_logits_shape": list(output.atomic_logits.shape),
        "atomic_probabilities_shape": list(output.atomic_probabilities.shape),
        "binary_probability_shape": list(output.binary_probability.shape),
        "image_grounding_shape": list(output.image_grounding_logits.shape),
        "text_grounding_shape": list(output.text_grounding_logits.shape),
        "valid_faces_per_sample": [int(v) for v in output.image_grounding_mask.sum(dim=1).detach().cpu().tolist()],
        "valid_full_tokens_per_sample": [int(v) for v in output.text_grounding_mask.sum(dim=1).detach().cpu().tolist()],
        "noisy_or_max_abs_error": float(noisy_or_error.detach().cpu()),
        "mean_p_fs": float(output.atomic_probabilities[:, 0].detach().float().mean().cpu()),
        "mean_p_fa": float(output.atomic_probabilities[:, 1].detach().float().mean().cpu()),
        "mean_p_ts": float(output.atomic_probabilities[:, 2].detach().float().mean().cpu()),
        "mean_p_ta": float(output.atomic_probabilities[:, 3].detach().float().mean().cpu()),
        "mean_p_fake": float(output.binary_probability.detach().float().mean().cpu()),
        "losses": losses.as_dict(),
    }


def _assert_gradient_connectivity(module: torch.nn.Module) -> None:
    gradient_tensors = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradient_tensors:
        raise RuntimeError("No gradients reached Phase-5 trainable chain")
    if not all(torch.isfinite(grad).all() for grad in gradient_tensors):
        raise FloatingPointError("Non-finite gradient in Phase-5 trainable chain")


if __name__ == "__main__":
    raise SystemExit(main())
