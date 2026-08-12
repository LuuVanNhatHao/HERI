"""Inspect Phase-4 graph construction without printing raw edge tensors."""

from __future__ import annotations

import json
import sys
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
from src.graph import HypothesisEvidenceGraphBuilder, HypothesisNodeInitializer
from src.graph.validation import graph_statistics, validate_graph_batch
from src.models.baselines.factory import build_backbones
from src.models.outputs import BackboneOutputs
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_graph_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_graph.py <config.yaml>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    config = load_yaml_config(config_path)
    validate_graph_config(config, mode="inspect")
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
    hypothesis_initializer = HypothesisNodeInitializer(config["graph"]).to(device)
    graph_builder = HypothesisEvidenceGraphBuilder(config["graph"])
    tokenizer = backbones["deberta"].tokenizer
    loader = create_dataloader(
        config,
        split=str(config["inspection"]["split"]),
        tokenizer=tokenizer,
        max_samples=int(config["inspection"]["max_samples"]),
        samples_per_class=config["inspection"].get("samples_per_class"),
    )

    records: list[dict[str, Any]] = []
    inspect_batches = int(config["inspection"]["inspect_batches"])
    iterator = tqdm(
        loader,
        total=min(inspect_batches, len(loader)),
        desc="Inspect graph",
        disable=not bool(config["progress"]["enabled"]),
    )
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if batch_index > inspect_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            with precision.autocast():
                evidence = evidence_extractor(_encode(backbones, batch), EvidenceStructuralMetadata.from_batch(batch))
                hypotheses = hypothesis_initializer(evidence.visual_semantic.features.shape[0])
                graph = graph_builder.build(evidence, hypotheses)
            validate_graph_batch(graph)
            stats = graph_statistics(graph)
            records.append({"batch_index": batch_index, "graph_statistics": stats})
            _print_summary(batch_index, stats, config)
    (output_root / "graph_inspection.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
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


def _print_summary(batch_index: int, stats: dict[str, Any], config: dict[str, Any]) -> None:
    print(f"batch_index: {batch_index}")
    print(f"batch_size: {stats['batch_size']}")
    print(f"node_counts: {stats['node_counts']}")
    print(f"edge_counts: {stats['edge_counts']}")
    print(f"hypothesis_tensor_shape: [{stats['batch_size']}, 4, {stats['hidden_dim']}]")
    print(f"graph_layers: {config['graph']['layers']}")
    print(f"heads: {config['graph']['heads']}")
    print(f"hidden_dim: {config['graph']['hidden_dim']}")
    print(f"feedback_enabled: {config['graph']['reasoning']['hypothesis_feedback']}")
    print(f"reasoning_cycles: {config['graph']['reasoning']['cycles']}")


if __name__ == "__main__":
    raise SystemExit(main())
