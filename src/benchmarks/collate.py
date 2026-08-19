"""Collation for task-aware binary HERA benchmark samples."""

from __future__ import annotations

from typing import Any

import torch


def binary_benchmark_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable text/face tensors while preserving signal availability."""

    if not batch:
        raise ValueError("Cannot collate an empty binary benchmark batch")
    pad_ids = {int(item["text_pad_token_id"]) for item in batch}
    if len(pad_ids) != 1:
        raise ValueError(f"Mixed tokenizer pad IDs in one batch: {sorted(pad_ids)}")
    pad_id = next(iter(pad_ids))
    return {
        "id": [item["id"] for item in batch],
        "split": [item["split"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
        "image_content_mask": torch.stack(
            [item["image_content_mask"] for item in batch]
        ),
        "image_path": [item["image_path"] for item in batch],
        "annotation_image_path": [item["annotation_image_path"] for item in batch],
        "image_transform_meta": [item["image_transform_meta"] for item in batch],
        "text": [item["text"] for item in batch],
        "words": [item["words"] for item in batch],
        "text_input_ids": _pad_1d(
            [item["text_input_ids"] for item in batch], pad_id, torch.long
        ),
        "text_attention_mask": _pad_1d(
            [item["text_attention_mask"] for item in batch], False, torch.bool
        ),
        "text_word_ids": _pad_1d(
            [item["text_word_ids"] for item in batch], -1, torch.long
        ),
        "text_grounding_mask": _pad_1d(
            [item["text_grounding_mask"] for item in batch], False, torch.bool
        ),
        "candidate_face_boxes": _pad_2d(
            [item["candidate_face_boxes"] for item in batch],
            width=4,
            dtype=torch.float32,
        ),
        "candidate_face_mask": _pad_1d(
            [item["candidate_face_mask"] for item in batch], False, torch.bool
        ),
        "binary_target": torch.stack([item["binary_target"] for item in batch]),
        "binary_available": torch.stack([item["binary_available"] for item in batch]),
        "subtype_target": torch.stack([item["subtype_target"] for item in batch]),
        "subtype_available": torch.stack([item["subtype_available"] for item in batch]),
        "text_label_target": torch.stack([item["text_label_target"] for item in batch]),
        "text_label_available": torch.stack(
            [item["text_label_available"] for item in batch]
        ),
        "metadata": [item["metadata"] for item in batch],
    }


def _pad_1d(
    tensors: list[torch.Tensor],
    pad_value: int | float | bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    length = max((int(tensor.shape[0]) for tensor in tensors), default=0)
    result = torch.full((len(tensors), length), pad_value, dtype=dtype)
    for row, tensor in enumerate(tensors):
        result[row, : tensor.shape[0]] = tensor.to(dtype=dtype)
    return result


def _pad_2d(
    tensors: list[torch.Tensor], *, width: int, dtype: torch.dtype
) -> torch.Tensor:
    length = max((int(tensor.shape[0]) for tensor in tensors), default=0)
    result = torch.zeros((len(tensors), length, width), dtype=dtype)
    for row, tensor in enumerate(tensors):
        if tensor.numel():
            result[row, : tensor.shape[0], :] = tensor.to(dtype=dtype)
    return result
