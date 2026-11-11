"""Shared encoder utilities for pretrained wrappers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


_BLOCK_PATH_TERMS = ("layer", "layers", "block", "blocks", "layer_chunks", "block_chunks")
_BLOCK_CLASS_TERMS = ("Layer", "Block", "Encoder")


def normalize_image_tensor(
    image: torch.Tensor,
    *,
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    """Apply model-specific channel normalization to [0,1] images."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image tensor [B,3,H,W], got {tuple(image.shape)}")
    device = image.device
    dtype = image.dtype
    mean_tensor = torch.tensor(list(mean), device=device, dtype=dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(list(std), device=device, dtype=dtype).view(1, 3, 1, 1)
    if torch.any(std_tensor <= 0):
        raise ValueError("Normalization std values must be positive")
    return (image - mean_tensor) / std_tensor


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool tokens while respecting a boolean mask."""

    if mask is None:
        return tokens.mean(dim=1)
    mask_f = mask.to(tokens.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (tokens * mask_f).sum(dim=1) / denom


def pixel_mask_to_patch_mask(
    pixel_mask: torch.Tensor | None,
    *,
    grid_height: int,
    grid_width: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Project a letterbox content mask onto a backbone patch grid.

    A patch remains valid when it overlaps at least one content pixel. This
    keeps boundary patches while excluding patches made entirely of padding.
    """

    if pixel_mask is None:
        return torch.ones((batch_size, grid_height * grid_width), dtype=torch.bool, device=device)
    if pixel_mask.ndim != 3 or pixel_mask.shape[0] != batch_size:
        raise ValueError("pixel_mask must be [B,H,W] and match the image batch")
    coverage = F.adaptive_avg_pool2d(
        pixel_mask.to(device=device, dtype=torch.float32).unsqueeze(1),
        output_size=(grid_height, grid_width),
    ).squeeze(1)
    patch_mask = coverage > 0.0
    if bool((~patch_mask.any(dim=(1, 2))).any()):
        raise ValueError("pixel_mask produced a sample with no valid image patches")
    return patch_mask.reshape(batch_size, grid_height * grid_width)


def infer_square_grid(num_tokens: int) -> tuple[int, int]:
    """Infer a square patch grid or fail clearly."""

    if num_tokens <= 0:
        raise ValueError(f"Cannot infer patch grid from non-positive token count: {num_tokens}")
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Cannot infer square patch grid from {num_tokens} patch tokens")
    return side, side


def freeze_module(module: nn.Module) -> None:
    """Freeze every parameter in a module."""

    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze every parameter in a module."""

    for parameter in module.parameters():
        parameter.requires_grad = True


def _module_sequence_has_parameters(blocks: list[nn.Module]) -> bool:
    return any(parameter.numel() > 0 for block in blocks for parameter in block.parameters())


def _flatten_module_sequence(sequence: nn.ModuleList | list[nn.Module]) -> list[nn.Module]:
    blocks: list[nn.Module] = []
    for item in sequence:
        if isinstance(item, nn.ModuleList):
            blocks.extend(_flatten_module_sequence(item))
        elif isinstance(item, list) and all(isinstance(child, nn.Module) for child in item):
            blocks.extend(_flatten_module_sequence(item))
        elif isinstance(item, nn.Module):
            blocks.append(item)
    return blocks


def _looks_like_transformer_blocks(path: str, blocks: list[nn.Module]) -> bool:
    if not blocks or not _module_sequence_has_parameters(blocks):
        return False
    normalized = path.lower()
    if any(term in normalized for term in _BLOCK_PATH_TERMS):
        return True
    return any(any(term in block.__class__.__name__ for term in _BLOCK_CLASS_TERMS) for block in blocks)


def _deduplicate_block_lists(candidates: list[list[nn.Module]]) -> list[list[nn.Module]]:
    deduplicated: list[list[nn.Module]] = []
    seen: set[tuple[int, ...]] = set()
    for blocks in candidates:
        key = tuple(id(block) for block in blocks)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(blocks)
    return deduplicated


def _candidate_block_lists(module: nn.Module) -> list[list[nn.Module]]:
    candidates: list[list[nn.Module]] = []
    paths = (
        "encoder.layer",
        "encoder.layers",
        "encoder.layer_chunks",
        "encoder.block_chunks",
        "layer",
        "layers",
        "blocks",
        "block_chunks",
        "vision_model.encoder.layers",
        "vision_model.encoder.layer",
        "vision_model.encoder.layer_chunks",
        "text_model.encoder.layers",
        "text_model.encoder.layer",
        "text_model.encoder.layer_chunks",
        "model.encoder.layers",
        "model.encoder.layer",
        "model.encoder.layer_chunks",
        "backbone.encoder.layers",
        "backbone.encoder.layer",
        "backbone.encoder.layer_chunks",
    )
    for dotted_path in paths:
        current: Any = module
        ok = True
        for part in dotted_path.split("."):
            if not hasattr(current, part):
                ok = False
                break
            current = getattr(current, part)
        if ok and isinstance(current, (nn.ModuleList, list)) and len(current) > 0:
            blocks = _flatten_module_sequence(current)
            if _looks_like_transformer_blocks(dotted_path, blocks):
                candidates.append(blocks)
    if candidates:
        return _deduplicate_block_lists(candidates)

    recursive: list[tuple[int, int, str, list[nn.Module]]] = []
    for path, child in module.named_modules():
        if not isinstance(child, nn.ModuleList) or len(child) == 0:
            continue
        blocks = _flatten_module_sequence(child)
        if not _looks_like_transformer_blocks(path, blocks):
            continue
        score = 0
        normalized = path.lower()
        if any(term in normalized for term in _BLOCK_PATH_TERMS):
            score += 10
        if "encoder" in normalized:
            score += 5
        if path.endswith(("layers", "layer", "blocks", "layer_chunks", "block_chunks")):
            score += 5
        recursive.append((score, len(blocks), path, blocks))
    if not recursive:
        return []
    recursive.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_score, best_len = recursive[0][0], recursive[0][1]
    best = [blocks for score, length, _, blocks in recursive if score == best_score and length == best_len]
    return _deduplicate_block_lists(best)


def unfreeze_last_blocks(module: nn.Module, count: int, *, stage_name: str) -> int:
    """Unfreeze the final ``count`` blocks using wrapper-level introspection."""

    if count <= 0:
        return 0
    block_lists = _candidate_block_lists(module)
    if not block_lists:
        raise ValueError(f"Cannot locate encoder blocks for partial unfreeze: {stage_name}")
    unfrozen = 0
    for blocks in block_lists:
        for block in blocks[-count:]:
            unfreeze_module(block)
            unfrozen += sum(parameter.numel() for parameter in block.parameters())
    if unfrozen == 0:
        raise ValueError(f"Requested partial unfreeze changed no parameters: {stage_name}")
    return unfrozen


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable
