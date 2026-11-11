"""Spatial metadata helpers for visual evidence."""

from __future__ import annotations

import torch

from heri.models.outputs import PatchGrid


def validate_patch_grid(*, token_count: int, patch_grid: PatchGrid, context: str) -> None:
    """Validate that a patch-token sequence matches its grid metadata."""

    expected = int(patch_grid.height) * int(patch_grid.width)
    if token_count != expected:
        raise ValueError(f"{context}: token count {token_count} != grid area {expected}")
    if patch_grid.height <= 0 or patch_grid.width <= 0:
        raise ValueError(f"{context}: patch grid height/width must be positive")
    if patch_grid.image_height <= 0 or patch_grid.image_width <= 0:
        raise ValueError(f"{context}: image height/width must be positive")


def patch_centers_and_boxes(
    patch_grid: PatchGrid,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return transformed-space and normalized patch centers/boxes."""

    x_edges = torch.linspace(0, patch_grid.image_width, patch_grid.width + 1, device=device, dtype=dtype)
    y_edges = torch.linspace(0, patch_grid.image_height, patch_grid.height + 1, device=device, dtype=dtype)
    x1 = x_edges[:-1]
    x2 = x_edges[1:]
    y1 = y_edges[:-1]
    y2 = y_edges[1:]
    grid_y1, grid_x1 = torch.meshgrid(y1, x1, indexing="ij")
    grid_y2, grid_x2 = torch.meshgrid(y2, x2, indexing="ij")
    boxes = torch.stack(
        [grid_x1.reshape(-1), grid_y1.reshape(-1), grid_x2.reshape(-1), grid_y2.reshape(-1)],
        dim=-1,
    )
    centers = torch.stack(
        [(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5],
        dim=-1,
    )
    norm_boxes = normalize_boxes(boxes, patch_grid=patch_grid)
    norm_centers = torch.stack(
        [centers[:, 0] / patch_grid.image_width, centers[:, 1] / patch_grid.image_height],
        dim=-1,
    ).clamp(0.0, 1.0)
    return centers, boxes, norm_centers, norm_boxes


def normalize_boxes(boxes: torch.Tensor, *, patch_grid: PatchGrid) -> torch.Tensor:
    """Normalize transformed-space boxes to [0,1]."""

    scale = boxes.new_tensor(
        [patch_grid.image_width, patch_grid.image_height, patch_grid.image_width, patch_grid.image_height]
    )
    return (boxes / scale).clamp(0.0, 1.0)


def gather_spatial(values: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    """Gather unbatched spatial metadata with padded -1 source indices."""

    if values.ndim != 2:
        raise ValueError(f"values must be [N,C], got {tuple(values.shape)}")
    if source_indices.ndim != 2:
        raise ValueError("source_indices must be [B,K]")
    batch, nodes = source_indices.shape
    channels = values.shape[-1]
    if nodes == 0:
        return values.new_zeros((batch, 0, channels))
    safe_indices = source_indices.clamp_min(0)
    gathered = values[safe_indices.reshape(-1)].reshape(batch, nodes, channels)
    return gathered.masked_fill(source_indices.unsqueeze(-1) < 0, 0.0)


def gather_batched(values: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
    """Gather [B,N,C] values with padded -1 source indices."""

    if values.ndim != 3:
        raise ValueError(f"values must be [B,N,C], got {tuple(values.shape)}")
    if source_indices.ndim != 2 or source_indices.shape[0] != values.shape[0]:
        raise ValueError("source_indices must be [B,K] aligned with values")
    batch, nodes = source_indices.shape
    channels = values.shape[-1]
    if nodes == 0:
        return values.new_zeros((batch, 0, channels))
    safe_indices = source_indices.clamp_min(0).unsqueeze(-1).expand(batch, nodes, channels)
    gathered = values.gather(1, safe_indices)
    return gathered.masked_fill(source_indices.unsqueeze(-1) < 0, 0.0)
