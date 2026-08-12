"""Deterministic, target-independent evidence budgeting."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .outputs import PAD_SOURCE_INDEX
from .spatial import gather_batched


@dataclass
class BudgetedFeatures:
    """Result of applying a graph-node budget."""

    features: torch.Tensor
    mask: torch.Tensor
    source_indices: torch.Tensor


def budget_features(
    *,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    max_nodes: int | None,
    strategy: str,
) -> BudgetedFeatures:
    """Select a deterministic subset while retaining original source indices."""

    if features.ndim != 3:
        raise ValueError(f"features must be [B,N,C], got {tuple(features.shape)}")
    if valid_mask.shape != features.shape[:2]:
        raise ValueError("valid_mask must be [B,N] aligned with features")
    mask = valid_mask.to(torch.bool)
    batch, _, channels = features.shape
    max_valid = int(mask.sum(dim=1).max().item()) if batch else 0
    if max_nodes is None:
        out_nodes = max_valid
    else:
        out_nodes = min(max_valid, int(max_nodes))
    if out_nodes < 0:
        raise ValueError("max_nodes must be non-negative")

    device = features.device
    source_indices = torch.full((batch, out_nodes), PAD_SOURCE_INDEX, dtype=torch.long, device=device)
    selected_mask = torch.zeros((batch, out_nodes), dtype=torch.bool, device=device)
    for batch_index in range(batch):
        valid_indices = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        chosen = _choose_indices(valid_indices, out_nodes, strategy=strategy)
        if chosen.numel() == 0:
            continue
        count = int(chosen.numel())
        source_indices[batch_index, :count] = chosen.to(torch.long)
        selected_mask[batch_index, :count] = True

    if out_nodes == 0:
        selected_features = features.new_zeros((batch, 0, channels))
    else:
        selected_features = gather_batched(features, source_indices)
    return BudgetedFeatures(features=selected_features, mask=selected_mask, source_indices=source_indices)


def _choose_indices(valid_indices: torch.Tensor, out_nodes: int, *, strategy: str) -> torch.Tensor:
    if out_nodes == 0 or valid_indices.numel() == 0:
        return valid_indices[:0]
    if valid_indices.numel() <= out_nodes:
        return valid_indices
    normalized = strategy.lower()
    if normalized in {"ordered", "all"}:
        return valid_indices[:out_nodes]
    if normalized in {"spatial_balanced", "sequence_balanced"}:
        return _linspace_indices(valid_indices, out_nodes)
    raise ValueError(f"Unsupported evidence budgeting strategy: {strategy!r}")


def _linspace_indices(valid_indices: torch.Tensor, out_nodes: int) -> torch.Tensor:
    positions = torch.linspace(
        0,
        valid_indices.numel() - 1,
        steps=out_nodes,
        device=valid_indices.device,
    ).round().to(torch.long)
    chosen = valid_indices[positions]
    deduped: list[int] = []
    seen: set[int] = set()
    for value in chosen.detach().cpu().tolist():
        if int(value) not in seen:
            seen.add(int(value))
            deduped.append(int(value))
    if len(deduped) < out_nodes:
        for value in valid_indices.detach().cpu().tolist():
            if int(value) not in seen:
                seen.add(int(value))
                deduped.append(int(value))
            if len(deduped) == out_nodes:
                break
    return torch.tensor(deduped, dtype=torch.long, device=valid_indices.device)
