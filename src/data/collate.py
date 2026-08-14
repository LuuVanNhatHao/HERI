"""Collate function for DGM4 variable-length data."""

from __future__ import annotations

from typing import Any

import torch


def _pad_2d_tensors(tensors: list[torch.Tensor], *, width: int, dtype: torch.dtype) -> torch.Tensor:
    max_len = max((tensor.shape[0] for tensor in tensors), default=0)
    padded = torch.zeros((len(tensors), max_len, width), dtype=dtype)
    for row, tensor in enumerate(tensors):
        if tensor.numel() == 0:
            continue
        padded[row, : tensor.shape[0], :] = tensor.to(dtype)
    return padded


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


def dgm4_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate DGM4 items into a structured batch with masks."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    face_boxes = [item["candidate_face_boxes"] for item in batch]
    face_masks = [item["candidate_face_mask"] for item in batch]
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
        "candidate_face_boxes": _pad_2d_tensors(face_boxes, width=4, dtype=torch.float32),
        "candidate_face_mask": _pad_1d_tensors(
            face_masks,
            pad_value=False,
            dtype=torch.bool,
        ),
        "fake_image_box_target": torch.stack(
            [item["fake_image_box_target"] for item in batch]
        ),
        "has_image_manipulation": torch.stack(
            [item["has_image_manipulation"] for item in batch]
        ),
        "manipulated_face_target_index": torch.stack(
            [item["manipulated_face_target_index"] for item in batch]
        ),
        "manipulated_face_target_iou": torch.stack(
            [item["manipulated_face_target_iou"] for item in batch]
        ),
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
        "fake_cls": [item["fake_cls"] for item in batch],
        "generator": [item["generator"] for item in batch],
        "image_source": [item["image_source"] for item in batch],
    }
