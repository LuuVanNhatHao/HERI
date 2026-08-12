"""Inspect a small DGM4 batch from a YAML config."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm.auto import tqdm

from src.compat.transformers_compat import TransformersCompatibilityError, load_fast_tokenizer
from src.data.loader import create_dataloader
from src.utils.config import get_key, load_yaml_config, validate_prepare_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def _maybe_load_tokenizer(config: dict[str, Any]) -> Any | None:
    if not bool(get_key(config, "tokenizer.enabled", False)):
        print("Tokenizer: disabled by YAML; using word-level Phase-1 alignment output.")
        return None
    try:
        return load_fast_tokenizer(
            name=str(get_key(config, "tokenizer.name")),
            use_fast=bool(get_key(config, "tokenizer.use_fast", True)),
            local_files_only=bool(get_key(config, "tokenizer.local_files_only", False)),
        )
    except TransformersCompatibilityError as exc:
        raise RuntimeError(str(exc)) from exc


def _shape(value: Any) -> str:
    if hasattr(value, "shape"):
        return str(tuple(value.shape))
    return f"list[{len(value)}]" if isinstance(value, list) else type(value).__name__


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_batch.py <config.yaml>", file=sys.stderr)
        return 2
    config = load_yaml_config(sys.argv[1])
    validate_prepare_config(config)
    setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    set_seed(int(config["project"]["seed"]))
    tokenizer = _maybe_load_tokenizer(config)
    split = str(get_key(config, "smoke.split", "train"))
    loader = create_dataloader(
        config,
        split=split,
        tokenizer=tokenizer,
        max_samples=int(get_key(config, "smoke.max_samples", 16)),
        samples_per_class=int(get_key(config, "smoke.samples_per_class", 1)),
    )
    inspect_batches = int(get_key(config, "smoke.inspect_batches", 1))
    for batch_index, batch in enumerate(
        tqdm(loader, total=min(inspect_batches, len(loader)), desc="Inspect DGM4 batch"),
        start=1,
    ):
        print(f"batch_index: {batch_index}")
        print(f"batch_size: {len(batch['id'])}")
        print(f"ids: {batch['id']}")
        print(f"image shape: {_shape(batch['image'])}")
        print(f"binary target shape: {_shape(batch['binary_target'])}")
        print(f"atomic target shape: {_shape(batch['atomic_target'])}")
        print(f"face-box shape: {_shape(batch['candidate_face_boxes'])}")
        print(f"face-box mask shape: {_shape(batch['candidate_face_mask'])}")
        print(f"text input shape: {_shape(batch['text_input_ids'])}")
        print(f"text attention mask shape: {_shape(batch['text_attention_mask'])}")
        print(f"text grounding shape: {_shape(batch['text_grounding_target'])}")
        print(f"text grounding mask shape: {_shape(batch['text_grounding_mask'])}")
        print(
            "image-manipulated samples: "
            f"{int(batch['has_image_manipulation'].sum().item())}"
        )
        print(
            "text-manipulated samples: "
            f"{int(batch['has_text_manipulation'].sum().item())}"
        )
        print(f"fake_cls: {batch['fake_cls']}")
        if batch_index >= inspect_batches:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
