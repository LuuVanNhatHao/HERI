"""Run a real Phase-1 to Phase-6 routing smoke check from YAML."""

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
from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor
from src.graph import HypothesisEvidenceGraphReasoner
from src.losses.hera_components import HERAMultitaskLoss
from src.models.baselines.factory import build_backbones
from src.models.heads.noisy_or import noisy_or_from_probabilities
from src.models.hera import HERADecisionModule
from src.models.outputs import BackboneOutputs
from src.routing import EvidenceReliabilityRouter, RoutingDiagnostics, summarize_routed_batch
from src.training.optimizer import ensure_no_uninitialized_parameters
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_routing_config
from src.utils.environment import collect_environment_metadata
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/routing_smoke.py <config.yaml>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    config = load_yaml_config(config_path)
    validate_routing_config(config, mode="smoke")
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
    router = EvidenceReliabilityRouter.from_config(config).to(device)
    reasoner = HypothesisEvidenceGraphReasoner.from_config(config).to(device)
    decision = HERADecisionModule.from_config(config).to(device)
    loss_fn = HERAMultitaskLoss(config["loss"])
    diagnostics = RoutingDiagnostics(
        low_threshold=float(config["routing"]["diagnostics"]["low_threshold"]),
        high_threshold=float(config["routing"]["diagnostics"]["high_threshold"]),
    )
    tokenizer = backbones["deberta"].tokenizer
    loader = create_dataloader(
        config,
        split=str(config["smoke"]["split"]),
        tokenizer=tokenizer,
        max_samples=int(config["smoke"]["max_samples"]),
        samples_per_class=config["smoke"].get("samples_per_class"),
    )

    backward = bool(config["smoke"].get("differentiability", False))
    trainable_chain = torch.nn.ModuleList([evidence_extractor, router, reasoner, decision])
    evidence_extractor.train(backward)
    router.train(backward)
    reasoner.train(backward)
    decision.train(backward)
    inspect_batches = int(config["smoke"]["inspect_batches"])
    records: list[dict[str, Any]] = []
    optimizer: torch.optim.Optimizer | None = None
    start = time.perf_counter()
    iterator = tqdm(
        loader,
        total=min(inspect_batches, len(loader)),
        desc="Routing smoke",
        disable=not bool(config["progress"]["enabled"]),
    )
    for batch_index, raw_batch in enumerate(iterator, start=1):
        if batch_index > inspect_batches:
            break
        batch = move_batch_to_device(raw_batch, device)
        with torch.set_grad_enabled(backward):
            with precision.autocast():
                evidence = evidence_extractor(_encode(backbones, batch), EvidenceStructuralMetadata.from_batch(batch))
                routed = router(evidence)
                graph_output = reasoner(routed.as_graph_evidence())
                decision_output = decision(graph_output)
                losses = loss_fn(decision_output, batch)
        if backward:
            if optimizer is None:
                ensure_no_uninitialized_parameters(trainable_chain, context="Phase-6 routing smoke optimizer creation")
                optimizer = torch.optim.AdamW(trainable_chain.parameters(), lr=float(config["smoke"]["projection_lr"]))
            optimizer.zero_grad(set_to_none=True)
            precision.backward(losses.total)
            _assert_router_gradients(router)
            precision.optimizer_step(optimizer)
        if not torch.isfinite(losses.total):
            raise FloatingPointError("Non-finite Phase-6 routing smoke loss")
        records.append(_diagnostics(batch_index, routed, decision_output, losses, diagnostics=diagnostics))
        print(json.dumps(records[-1], indent=2))

    ensure_no_uninitialized_parameters(trainable_chain, context="Phase-6 routing smoke completion")
    report = {
        "device": str(device),
        "precision_requested": precision.requested,
        "precision_actual": precision.actual,
        "runtime_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "routing": {
            "enabled": bool(config["routing"]["enabled"]),
            "mode": str(config["routing"]["mode"]),
            "estimator_hidden_dim": int(config["routing"]["estimator"]["hidden_dim"]),
        },
        "benchmark_performance_claim": False,
        "batches": records,
    }
    (output_root / "routing_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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


def _diagnostics(
    batch_index: int,
    routed: Any,
    decision_output: Any,
    losses: Any,
    *,
    diagnostics: RoutingDiagnostics,
) -> dict[str, Any]:
    reference_fake = noisy_or_from_probabilities(decision_output.atomic_probabilities.float())
    noisy_or_error = torch.max(torch.abs(reference_fake - decision_output.binary_probability.float()))
    return {
        "batch_index": batch_index,
        "routing_summary": summarize_routed_batch(routed, diagnostics=diagnostics),
        "hypothesis_shape": list(decision_output.hypothesis_features.shape),
        "atomic_logits_shape": list(decision_output.atomic_logits.shape),
        "image_grounding_shape": list(decision_output.image_grounding_logits.shape),
        "text_grounding_shape": list(decision_output.text_grounding_logits.shape),
        "noisy_or_max_abs_error": float(noisy_or_error.detach().cpu()),
        "losses": losses.as_dict(),
    }


def _assert_router_gradients(router: EvidenceReliabilityRouter) -> None:
    if not router.enabled:
        return
    gradients = [
        parameter.grad
        for parameter in router.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("No gradients reached Phase-6 reliability router")
    if not all(torch.isfinite(grad).all() for grad in gradients):
        raise FloatingPointError("Non-finite gradient in Phase-6 reliability router")


if __name__ == "__main__":
    raise SystemExit(main())
