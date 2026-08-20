"""Inspect Phase-5 decision outputs without printing raw tensors."""

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
from src.graph import HypothesisEvidenceGraphReasoner
from src.models.baselines.factory import build_backbones
from src.models.heads.noisy_or import noisy_or_from_probabilities
from src.models.hera import HERADecisionModule
from src.models.outputs import BackboneOutputs
from src.training.precision import PrecisionManager
from src.utils.config import dump_yaml, load_yaml_config, validate_decision_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_decision.py <config.yaml>", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1])
    config = load_yaml_config(config_path)
    validate_decision_config(config, mode="inspect")
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
        desc="Inspect decision",
        disable=not bool(config["progress"]["enabled"]),
    )
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(iterator, start=1):
            if batch_index > inspect_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            with precision.autocast():
                evidence = evidence_extractor(_encode(backbones, batch), EvidenceStructuralMetadata.from_batch(batch))
                graph_output = reasoner(evidence)
                output = decision(graph_output)
            record = _summary(batch_index, output, batch)
            records.append(record)
            _print_summary(record)
    (output_root / "decision_inspection.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
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


def _summary(batch_index: int, output: Any, batch: dict[str, Any]) -> dict[str, Any]:
    reference_fake = noisy_or_from_probabilities(output.atomic_probabilities.float())
    word_ids = output.text_word_ids
    valid_word_ids = word_ids[output.text_grounding_mask] if word_ids is not None else torch.empty(0, dtype=torch.long)
    return {
        "batch_index": batch_index,
        "hypothesis_feature_shape": list(output.hypothesis_features.shape),
        "atomic_logits_shape": list(output.atomic_logits.shape),
        "atomic_probabilities_shape": list(output.atomic_probabilities.shape),
        "binary_probability_shape": list(output.binary_probability.shape),
        "noisy_or_max_abs_error": float(torch.max(torch.abs(reference_fake - output.binary_probability.float())).cpu()),
        "image_grounding_shape": list(output.image_grounding_logits.shape),
        "valid_image_patches_per_sample": [
            int(v) for v in output.image_grounding_mask.sum(dim=1).detach().cpu().tolist()
        ],
        "text_grounding_shape": list(output.text_grounding_logits.shape),
        "valid_full_tokens_per_sample": [int(v) for v in output.text_grounding_mask.sum(dim=1).detach().cpu().tolist()],
        "word_id_min": int(valid_word_ids.min().cpu()) if valid_word_ids.numel() else None,
        "word_id_max": int(valid_word_ids.max().cpu()) if valid_word_ids.numel() else None,
        "image_applicability_fs_fa": batch["atomic_target"][:, :2].detach().cpu().int().tolist(),
        "text_applicability_ts_ta": batch["atomic_target"][:, 2:4].detach().cpu().int().tolist(),
        "mean_atomic_probabilities": [float(v) for v in output.atomic_probabilities.detach().float().mean(dim=0).cpu().tolist()],
        "mean_p_fake": float(output.binary_probability.detach().float().mean().cpu()),
    }


def _print_summary(record: dict[str, Any]) -> None:
    print(f"batch_index: {record['batch_index']}")
    print(f"hypothesis_feature_shape: {record['hypothesis_feature_shape']}")
    print(f"atomic_logits_shape: {record['atomic_logits_shape']}")
    print(f"atomic_probabilities_shape: {record['atomic_probabilities_shape']}")
    print(f"binary_probability_shape: {record['binary_probability_shape']}")
    print(f"noisy_or_max_abs_error: {record['noisy_or_max_abs_error']:.8f}")
    print(f"image_grounding_shape: {record['image_grounding_shape']}")
    print(f"valid_image_patches_per_sample: {record['valid_image_patches_per_sample']}")
    print(f"text_grounding_shape: {record['text_grounding_shape']}")
    print(f"valid_full_tokens_per_sample: {record['valid_full_tokens_per_sample']}")
    print(f"word_id_range: {record['word_id_min']}..{record['word_id_max']}")
    print(f"image_applicability_fs_fa: {record['image_applicability_fs_fa']}")
    print(f"text_applicability_ts_ta: {record['text_applicability_ts_ta']}")
    print(f"mean_atomic_probabilities: {record['mean_atomic_probabilities']}")
    print(f"mean_p_fake: {record['mean_p_fake']:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
