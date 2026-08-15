"""Inspect Phase-6 routing reliability diagnostics without raw tensors."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import create_dataloader
from src.evaluation.evaluator import move_batch_to_device
from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor
from src.models.baselines.factory import build_backbones
from src.models.outputs import BackboneOutputs
from src.routing import EvidenceReliabilityRouter, RoutingDiagnostics, summarize_routed_batch
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_routing_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_routing.py <config.yaml>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    config = load_yaml_config(config_path)
    validate_routing_config(config, mode="inspect")
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
    router.eval()
    diagnostics = RoutingDiagnostics(
        low_threshold=float(config["routing"]["diagnostics"]["low_threshold"]),
        high_threshold=float(config["routing"]["diagnostics"]["high_threshold"]),
    )
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
        desc="Inspect routing",
        disable=not bool(config["progress"]["enabled"]),
    )
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if batch_index > inspect_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            with precision.autocast():
                evidence = evidence_extractor(_encode(backbones, batch), EvidenceStructuralMetadata.from_batch(batch))
                routed = router(evidence)
            summary = summarize_routed_batch(routed, diagnostics=diagnostics)
            record = {"batch_index": batch_index, "batch_size": int(evidence.visual_semantic.features.shape[0]), "summary": summary}
            records.append(record)
            _print_summary(record)
    (output_root / "routing_inspection.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
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


def _print_summary(record: dict[str, Any]) -> None:
    summary = record["summary"]
    print(f"batch_index: {record['batch_index']}")
    print(f"batch_size: {record['batch_size']}")
    print(f"router_enabled: {summary['router']['enabled']}")
    print(f"routing_mode: {summary['router']['mode']}")
    print(f"hard_routing_enabled: {summary['router']['hard_eval_enabled']}")
    for family, family_summary in summary["families"].items():
        print(f"{family}:")
        print(f"  feature_shape: {family_summary['feature_shape']}")
        print(f"  valid_nodes_per_sample: {family_summary['valid_nodes_per_sample']}")
        print(f"  reliability_mean: {family_summary['reliability_mean']:.6f}")
        print(f"  reliability_std: {family_summary['reliability_std']:.6f}")
        print(f"  reliability_min: {family_summary['reliability_min']:.6f}")
        print(f"  reliability_max: {family_summary['reliability_max']:.6f}")
        print(f"  fraction_below_low_threshold: {family_summary['fraction_below_low_threshold']:.6f}")
        print(f"  fraction_above_high_threshold: {family_summary['fraction_above_high_threshold']:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
