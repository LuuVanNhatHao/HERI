"""Inspect Phase-3 evidence batches without printing raw tensors."""

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
from src.evidence.validation import summarize_evidence_batch, validate_evidence_batch
from src.models.baselines.factory import build_backbones
from src.models.outputs import BackboneOutputs
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_evidence_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_evidence.py <config.yaml>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    config = load_yaml_config(config_path)
    validate_evidence_config(config, mode="inspect")
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
    extractor = HeterogeneousEvidenceExtractor.from_config(config).to(device)
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
    progress = bool(config["progress"]["enabled"])
    iterator = tqdm(loader, total=min(inspect_batches, len(loader)), desc="Inspect evidence", disable=not progress)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if batch_index > inspect_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            with precision.autocast():
                encoder_outputs = _encode(backbones, batch)
                evidence = extractor(encoder_outputs, EvidenceStructuralMetadata.from_batch(batch))
            validate_evidence_batch(evidence)
            summary = summarize_evidence_batch(evidence)
            records.append({"batch_index": batch_index, "summary": summary})
            _print_summary(batch_index, summary)
    (output_root / "evidence_inspection.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
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


def _print_summary(batch_index: int, summary: dict[str, Any]) -> None:
    print(f"batch_index: {batch_index}")
    for family in ("visual_semantic", "visual_forensic", "face", "text_alignment", "text_context"):
        family_summary = summary[family]
        print(f"{family}:")
        print(f"  feature_shape: {family_summary['feature_shape']}")
        print(f"  valid_nodes_per_sample: {family_summary['valid_nodes_per_sample']}")
        if family_summary["spatial_boxes_shape"] is not None:
            print(f"  spatial_boxes_shape: {family_summary['spatial_boxes_shape']}")
        if family_summary["word_ids_shape"] is not None:
            print(f"  word_ids_shape: {family_summary['word_ids_shape']}")
    print("cross_modal:")
    print(f"  method: {summary['cross_modal']['method']}")
    print(f"  similarity_shape: {summary['cross_modal']['similarity_shape']}")


if __name__ == "__main__":
    raise SystemExit(main())
