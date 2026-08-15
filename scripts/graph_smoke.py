"""Run a real-backbone + real-HGT Phase-4 smoke check from YAML."""

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
from src.graph.validation import validate_hypothesis_output
from src.models.baselines.factory import build_backbones
from src.models.outputs import BackboneOutputs
from src.training.optimizer import ensure_no_uninitialized_parameters
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_graph_config
from src.utils.environment import collect_environment_metadata
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/graph_smoke.py <config.yaml>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    config = load_yaml_config(config_path)
    validate_graph_config(config, mode="smoke")
    setup_logging(str(config["logging"]["level"]))
    set_seed(int(config["project"]["seed"]))
    output_root = Path(str(config["output"]["root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_root / "effective_config.yaml")

    device = torch.device(str(config["runtime"]["device"]))
    precision = PrecisionManager(precision=str(config["runtime"]["precision"]), device=device)
    reasoner = HypothesisEvidenceGraphReasoner.from_config(config).to(device)
    backbones = build_backbones(config)
    for module in backbones.values():
        module.to(device)
        module.eval()
    evidence_extractor = HeterogeneousEvidenceExtractor.from_config(config).to(device)
    tokenizer = backbones["deberta"].tokenizer
    loader = create_dataloader(
        config,
        split=str(config["smoke"]["split"]),
        tokenizer=tokenizer,
        max_samples=int(config["smoke"]["max_samples"]),
        samples_per_class=config["smoke"].get("samples_per_class"),
    )

    backward = bool(config["smoke"].get("differentiability", False))
    inspect_batches = int(config["smoke"]["inspect_batches"])
    optimizer: torch.optim.Optimizer | None = None
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    iterator = tqdm(
        loader,
        total=min(inspect_batches, len(loader)),
        desc="Graph smoke",
        disable=not bool(config["progress"]["enabled"]),
    )
    for batch_index, raw_batch in enumerate(iterator, start=1):
        if batch_index > inspect_batches:
            break
        batch = move_batch_to_device(raw_batch, device)
        with torch.set_grad_enabled(backward):
            with precision.autocast():
                evidence = evidence_extractor(_encode(backbones, batch), EvidenceStructuralMetadata.from_batch(batch))
                output = reasoner(evidence)
                validate_hypothesis_output(output)
                if backward:
                    scalar = output.hypothesis_features.float().pow(2).mean()
        if backward:
            if optimizer is None:
                ensure_no_uninitialized_parameters(reasoner, context="Phase-4 graph smoke optimizer creation")
                optimizer = torch.optim.AdamW(reasoner.parameters(), lr=float(config["smoke"].get("projection_lr", 1e-4)))
            optimizer.zero_grad(set_to_none=True)
            precision.backward(scalar)
            precision.optimizer_step(optimizer)
        record = {
            "batch_index": batch_index,
            "hypothesis_shape": list(output.hypothesis_features.shape),
            "graph_statistics": output.graph_statistics,
        }
        records.append(record)
        print(json.dumps(record, indent=2))
    ensure_no_uninitialized_parameters(reasoner, context="Phase-4 graph smoke completion")
    report = {
        "device": str(device),
        "precision_requested": precision.requested,
        "precision_actual": precision.actual,
        "runtime_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "graph": {
            "type": config["graph"]["type"],
            "hidden_dim": config["graph"]["hidden_dim"],
            "layers": config["graph"]["layers"],
            "heads": config["graph"]["heads"],
            "cycles": config["graph"]["reasoning"]["cycles"],
            "hypothesis_feedback": config["graph"]["reasoning"]["hypothesis_feedback"],
        },
        "batches": records,
    }
    (output_root / "graph_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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


if __name__ == "__main__":
    raise SystemExit(main())
