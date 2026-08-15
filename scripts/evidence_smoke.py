"""Run a real-backbone Phase-3 evidence smoke check from YAML."""

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
from src.evidence.validation import summarize_evidence_batch, validate_evidence_batch
from src.models.baselines.factory import build_backbones
from src.models.outputs import BackboneOutputs
from src.training.optimizer import ensure_no_uninitialized_parameters
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_evidence_config
from src.utils.environment import collect_environment_metadata
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/evidence_smoke.py <config.yaml>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    config = load_yaml_config(config_path)
    validate_evidence_config(config, mode="smoke")
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
        split=str(config["smoke"]["split"]),
        tokenizer=tokenizer,
        max_samples=int(config["smoke"]["max_samples"]),
        samples_per_class=config["smoke"].get("samples_per_class"),
    )

    backward = bool(config["smoke"].get("differentiability", False))
    inspect_batches = int(config["smoke"]["inspect_batches"])
    progress = bool(config["progress"]["enabled"])
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    iterator = tqdm(loader, total=min(inspect_batches, len(loader)), desc="Evidence smoke", disable=not progress)
    optimizer: torch.optim.Optimizer | None = None
    for batch_index, raw_batch in enumerate(iterator, start=1):
        if batch_index > inspect_batches:
            break
        batch = move_batch_to_device(raw_batch, device)
        with torch.set_grad_enabled(backward):
            with precision.autocast():
                encoder_outputs = _encode(backbones, batch)
                evidence = extractor(
                    encoder_outputs,
                    EvidenceStructuralMetadata.from_batch(batch),
                )
                validate_evidence_batch(evidence)
                if backward:
                    scalar = _finite_projection_scalar(evidence)
        if backward:
            if optimizer is None:
                ensure_no_uninitialized_parameters(extractor, context="Phase-3 evidence smoke optimizer creation")
                optimizer = torch.optim.AdamW(extractor.parameters(), lr=float(config["smoke"].get("projection_lr", 1e-4)))
            optimizer.zero_grad(set_to_none=True)
            precision.backward(scalar)
            precision.optimizer_step(optimizer)
        summary = summarize_evidence_batch(evidence)
        records.append({"batch_index": batch_index, "summary": summary})
        print(json.dumps({"batch_index": batch_index, "summary": summary}, indent=2))

    ensure_no_uninitialized_parameters(extractor, context="Phase-3 evidence smoke completion")
    runtime = time.perf_counter() - start
    report = {
        "script": str(config_path),
        "device": str(device),
        "precision_requested": precision.requested,
        "precision_actual": precision.actual,
        "model_identifiers": {
            "siglip2": str(config["backbones"]["siglip2"]["name"]),
            "dinov3": str(config["backbones"]["dinov3"]["name"]),
            "deberta": str(config["backbones"]["deberta"]["name"]),
        },
        "runtime_seconds": runtime,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "batches": records,
    }
    (output_root / "evidence_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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


def _finite_projection_scalar(evidence: Any) -> torch.Tensor:
    values = []
    for evidence_set in evidence.families().values():
        if bool(evidence_set.mask.any()):
            values.append(evidence_set.features[evidence_set.mask].float().pow(2).mean())
    if not values:
        raise RuntimeError("Differentiability smoke found no valid evidence nodes")
    scalar = torch.stack(values).sum()
    if not torch.isfinite(scalar):
        raise FloatingPointError("Non-finite evidence differentiability scalar")
    return scalar


if __name__ == "__main__":
    raise SystemExit(main())
