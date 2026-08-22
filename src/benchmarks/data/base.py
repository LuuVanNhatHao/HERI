"""Common item construction for multimodal binary benchmark datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import image_content_mask, preprocess_image


@dataclass(frozen=True)
class BinaryBenchmarkRecord:
    """Raw adapter-neutral sample before image/text tensorization."""

    sample_id: str
    text: str
    image: Image.Image
    image_reference: str
    binary_target: float
    subtype_target: int = -100
    subtype_available: bool = False
    text_label_target: float = 0.0
    text_label_available: bool = False
    metadata: Mapping[str, Any] | None = None
    text_parts: Mapping[str, str] | None = None


class BinaryBenchmarkDataset(Dataset[dict[str, Any]], ABC):
    """Base dataset that emits the structural HERA batch contract without fake targets."""

    def __init__(
        self,
        *,
        split: str,
        tokenizer: Any,
        tokenizer_max_length: int,
        image_size: int,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split!r}")
        if tokenizer is None or not hasattr(tokenizer, "__call__"):
            raise TypeError("A callable fast tokenizer is required")
        self.split = split
        self.tokenizer = tokenizer
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.image_size = int(image_size)
        if self.tokenizer_max_length <= 0 or self.image_size <= 0:
            raise ValueError("tokenizer_max_length and image_size must be positive")

    @abstractmethod
    def raw_record(self, index: int) -> BinaryBenchmarkRecord:
        """Load one raw record and an owned PIL image."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.raw_record(index)
        try:
            image_tensor, transform_meta = preprocess_image(
                record.image,
                target_size=self.image_size,
            )
        finally:
            record.image.close()
        text = str(record.text).strip()
        if not text:
            raise ValueError(
                f"Empty benchmark text: split={self.split} id={record.sample_id}"
            )
        encoded = self.encode_record_text(record)
        encoding_audit = dict(encoded.get("audit", {}))
        metadata = dict(record.metadata or {})
        if encoding_audit:
            metadata["text_encoding"] = encoding_audit
        return {
            "id": str(record.sample_id),
            "split": self.split,
            "image": image_tensor,
            "image_content_mask": image_content_mask(transform_meta),
            "image_path": record.image_reference,
            "annotation_image_path": record.image_reference,
            "image_transform_meta": transform_meta.to_dict(),
            "text": text,
            "words": text.split(),
            "text_input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "text_attention_mask": torch.tensor(
                encoded["attention_mask"], dtype=torch.bool
            ),
            "text_word_ids": torch.tensor(encoded["word_ids"], dtype=torch.long),
            # This is a structural valid-token mask, not a grounding target.
            "text_grounding_mask": torch.tensor(
                encoded["grounding_mask"], dtype=torch.bool
            ),
            "binary_target": torch.tensor(
                float(record.binary_target), dtype=torch.float32
            ),
            "binary_available": torch.tensor(True, dtype=torch.bool),
            "subtype_target": torch.tensor(
                int(record.subtype_target), dtype=torch.long
            ),
            "subtype_available": torch.tensor(
                bool(record.subtype_available), dtype=torch.bool
            ),
            "text_label_target": torch.tensor(
                float(record.text_label_target), dtype=torch.float32
            ),
            "text_label_available": torch.tensor(
                bool(record.text_label_available), dtype=torch.bool
            ),
            "metadata": metadata,
            "text_pad_token_id": int(getattr(self.tokenizer, "pad_token_id", 0) or 0),
        }

    def encode_record_text(self, record: BinaryBenchmarkRecord) -> dict[str, Any]:
        """Encode one record; dataset adapters may override selection policy."""

        return _encode_text(
            self.tokenizer,
            record.text,
            max_length=self.tokenizer_max_length,
        )


def _encode_text(
    tokenizer: Any, text: str, *, max_length: int
) -> dict[str, list[int] | list[bool]]:
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    input_ids = _to_int_list(encoding["input_ids"])
    attention_mask = _to_int_list(encoding.get("attention_mask", [1] * len(input_ids)))
    if len(input_ids) != len(attention_mask):
        raise ValueError("Tokenizer input_ids and attention_mask lengths differ")
    word_ids_method = getattr(encoding, "word_ids", None)
    if not callable(word_ids_method):
        raise TypeError(
            "Binary benchmark tokenization requires a fast tokenizer with word_ids()"
        )
    word_ids_raw = word_ids_method()
    if len(word_ids_raw) != len(input_ids):
        raise ValueError("Tokenizer word_ids do not align with input_ids")
    word_ids = [-1 if value is None else int(value) for value in word_ids_raw]
    grounding_mask = [
        bool(mask) and word_id >= 0 for mask, word_id in zip(attention_mask, word_ids)
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_ids": word_ids,
        "grounding_mask": grounding_mask,
    }


def _to_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Per-item tokenization unexpectedly returned a batch")
        value = value[0]
    return [int(item) for item in value]
