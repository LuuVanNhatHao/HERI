"""Collate functions for variable-length multimodal manipulation data."""

from __future__ import annotations

from typing import Any

import torch


def _pad_1d_tensors(
    tensors: list[torch.Tensor],
    *,
    pad_value: int | float | bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_len = max((tensor.shape[0] for tensor in tensors), default=0)
    padded = torch.full((len(tensors), max_len), pad_value, dtype=dtype)
    for row, tensor in enumerate(tensors):
        if tensor.numel() == 0:
            continue
        padded[row, : tensor.shape[0]] = tensor.to(dtype)
    return padded


def multimodal_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate normalized DGM4/SAMM items into a structured batch."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    text_input_ids = [item["text_input_ids"] for item in batch]
    text_attention_masks = [item["text_attention_mask"] for item in batch]
    text_grounding_targets = [item["text_grounding_target"] for item in batch]
    text_grounding_masks = [item["text_grounding_mask"] for item in batch]
    text_word_ids = [item["text_word_ids"] for item in batch]
    fake_text_positions = [item["fake_text_pos"] for item in batch]

    return {
        "id": [item["id"] for item in batch],
        "split": [item["split"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
        "image_content_mask": torch.stack([item["image_content_mask"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
        "annotation_image_path": [item["annotation_image_path"] for item in batch],
        "image_transform_meta": [item["image_transform_meta"] for item in batch],
        "text": [item["text"] for item in batch],
        "words": [item["words"] for item in batch],
        "binary_target": torch.stack([item["binary_target"] for item in batch]),
        "atomic_target": torch.stack([item["atomic_target"] for item in batch]),
        "fake_image_box_target": torch.stack(
            [item["fake_image_box_target"] for item in batch]
        ),
        "has_image_manipulation": torch.stack(
            [item["has_image_manipulation"] for item in batch]
        ),
        "image_grounding_available": torch.stack(
            [item.get("image_grounding_available", torch.tensor(True)) for item in batch]
        ).to(torch.bool),
        "fake_text_pos": _pad_1d_tensors(
            fake_text_positions,
            pad_value=-1,
            dtype=torch.long,
        ),
        "fake_text_pos_mask": _pad_1d_tensors(
            [torch.ones_like(pos, dtype=torch.bool) for pos in fake_text_positions],
            pad_value=False,
            dtype=torch.bool,
        ),
        "text_input_ids": _pad_1d_tensors(text_input_ids, pad_value=0, dtype=torch.long),
        "text_attention_mask": _pad_1d_tensors(
            text_attention_masks,
            pad_value=False,
            dtype=torch.bool,
        ),
        "text_grounding_target": _pad_1d_tensors(
            text_grounding_targets,
            pad_value=-100,
            dtype=torch.long,
        ),
        "text_grounding_mask": _pad_1d_tensors(
            text_grounding_masks,
            pad_value=False,
            dtype=torch.bool,
        ),
        "text_word_ids": _pad_1d_tensors(
            text_word_ids,
            pad_value=-1,
            dtype=torch.long,
        ),
        "has_text_manipulation": torch.stack(
            [item["has_text_manipulation"] for item in batch]
        ),
        "text_grounding_available": torch.stack(
            [item.get("text_grounding_available", torch.tensor(True)) for item in batch]
        ).to(torch.bool),
        "fake_cls": [item["fake_cls"] for item in batch],
        "generator": [item["generator"] for item in batch],
        "image_source": [item["image_source"] for item in batch],
        "dataset_name": [item.get("dataset_name", "DGM4") for item in batch],
        "class_target": torch.tensor(
            [int(item.get("class_target", -1)) for item in batch], dtype=torch.long
        ),
        "class_names": list(batch[0].get("class_names", [])),
        "hypothesis_names": list(batch[0].get("hypothesis_names", [])),
    }


# Public compatibility alias retained for all existing DGM4 callers/tests.
dgm4_collate = multimodal_collate
