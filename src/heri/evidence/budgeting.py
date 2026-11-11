"""Deterministic, target-independent evidence budgeting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .outputs import PAD_SOURCE_INDEX
from .spatial import gather_batched


@dataclass
class BudgetedFeatures:
    """Result of applying a graph-node budget."""

    features: torch.Tensor
    mask: torch.Tensor
    source_indices: torch.Tensor
    selection_scores: torch.Tensor | None = None
    ungated_features: torch.Tensor | None = None


def budget_features(
    *,
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    max_nodes: int | None,
    strategy: str,
    selection_scores: torch.Tensor | None = None,
    eta: float = 0.5,
) -> BudgetedFeatures:
    """Select a deterministic subset while retaining original source indices."""

    if features.ndim != 3:
        raise ValueError(f"features must be [B,N,C], got {tuple(features.shape)}")
    if valid_mask.shape != features.shape[:2]:
        raise ValueError("valid_mask must be [B,N] aligned with features")
    mask = valid_mask.to(torch.bool)
    if selection_scores is not None:
        if selection_scores.shape != mask.shape:
            raise ValueError("selection_scores must be [B,N] aligned with features")
        if not torch.isfinite(selection_scores).all():
            raise FloatingPointError("selection_scores contains non-finite values")
    batch, _, channels = features.shape
    max_valid = int(mask.sum(dim=1).max().item()) if batch else 0
    if max_nodes is None:
        out_nodes = max_valid
    else:
        out_nodes = min(max_valid, int(max_nodes))
    if out_nodes < 0:
        raise ValueError("max_nodes must be non-negative")

    device = features.device
    source_indices = torch.full(
        (batch, out_nodes), PAD_SOURCE_INDEX, dtype=torch.long, device=device
    )
    selected_mask = torch.zeros((batch, out_nodes), dtype=torch.bool, device=device)
    for batch_index in range(batch):
        valid_indices = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        sample_scores = (
            None if selection_scores is None else selection_scores[batch_index]
        )
        chosen = _choose_indices(
            valid_indices,
            out_nodes,
            strategy=strategy,
            selection_scores=sample_scores,
            features=features[batch_index],
            eta=eta,
        )
        if chosen.numel() == 0:
            continue
        count = int(chosen.numel())
        source_indices[batch_index, :count] = chosen.to(torch.long)
        selected_mask[batch_index, :count] = True

    if out_nodes == 0:
        selected_features = features.new_zeros((batch, 0, channels))
    else:
        selected_features = gather_batched(features, source_indices)
    selected_scores = None
    if selection_scores is not None:
        selected_scores = gather_batched(
            selection_scores.unsqueeze(-1), source_indices
        ).squeeze(-1)
        selected_scores = selected_scores.masked_fill(~selected_mask, 0.0)
    return BudgetedFeatures(
        features=selected_features,
        mask=selected_mask,
        source_indices=source_indices,
        selection_scores=selected_scores,
    )


def _choose_indices(
    valid_indices: torch.Tensor,
    out_nodes: int,
    *,
    strategy: str,
    selection_scores: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    eta: float = 0.5,
) -> torch.Tensor:
    if out_nodes == 0 or valid_indices.numel() == 0:
        return valid_indices[:0]
    if valid_indices.numel() <= out_nodes:
        return valid_indices
    normalized = strategy.lower()
    if normalized in {"ordered", "all"}:
        return valid_indices[:out_nodes]
    if normalized in {"spatial_balanced", "sequence_balanced"}:
        return _linspace_indices(valid_indices, out_nodes)
    if normalized in {
        "saliency_spatial",
        "saliency_sequence",
        "forensic_saliency",
        "greedy_saliency_coverage",
    }:
        if selection_scores is None:
            raise ValueError(
                f"Budgeting strategy {strategy!r} requires selection_scores"
            )
        if features is None:
            raise ValueError(
                "Greedy saliency–coverage selection requires candidate features"
            )
        return greedy_saliency_coverage_indices(
            valid_indices, selection_scores, features, out_nodes, eta=eta
        )
    raise ValueError(f"Unsupported evidence budgeting strategy: {strategy!r}")


def _linspace_indices(valid_indices: torch.Tensor, out_nodes: int) -> torch.Tensor:
    positions = (
        torch.linspace(
            0,
            valid_indices.numel() - 1,
            steps=out_nodes,
            device=valid_indices.device,
        )
        .round()
        .to(torch.long)
    )
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


def _saliency_balanced_indices(
    valid_indices: torch.Tensor,
    selection_scores: torch.Tensor,
    out_nodes: int,
) -> torch.Tensor:
    """Mix high-importance nodes with uniform coverage, without using targets."""

    salient_count = max(1, out_nodes // 2)
    ranked = torch.argsort(
        selection_scores[valid_indices], descending=True, stable=True
    )
    salient = valid_indices[ranked[:salient_count]]
    coverage = _linspace_indices(valid_indices, out_nodes)
    chosen: list[int] = []
    seen: set[int] = set()
    for tensor in (salient, coverage, valid_indices):
        for value in tensor.detach().cpu().tolist():
            value = int(value)
            if value in seen:
                continue
            seen.add(value)
            chosen.append(value)
            if len(chosen) == out_nodes:
                return torch.tensor(
                    chosen, dtype=torch.long, device=valid_indices.device
                )
    return torch.tensor(chosen, dtype=torch.long, device=valid_indices.device)


def greedy_saliency_coverage_indices(
    valid_indices: torch.Tensor,
    selection_scores: torch.Tensor,
    features: torch.Tensor,
    budget: int,
    *,
    eta: float = 0.5,
) -> torch.Tensor:
    """Eq.13, with source-index order breaking exact score ties.

    Selection is discrete; the gathered continuous scores retain gradients for
    the saliency gate applied by ``LearnedEvidenceSelector``.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("evidence.selection_eta must be in [0,1]")
    if budget <= 0 or valid_indices.numel() == 0:
        return valid_indices[:0]
    with torch.no_grad():
        candidates = valid_indices.sort().values
        score = selection_scores[candidates].float()
        norm = torch.nn.functional.normalize(
            features[candidates].float(), dim=-1, eps=1e-8
        )
        similarity = norm @ norm.T
        chosen = [int(torch.argmax(score).item())]
        remaining = torch.ones(
            candidates.numel(), dtype=torch.bool, device=candidates.device
        )
        remaining[chosen[0]] = False
        best_similarity = similarity[:, chosen[0]]
        for _ in range(min(budget, candidates.numel()) - 1):
            objective = eta * score + (1.0 - eta) * (1.0 - best_similarity)
            objective = objective.masked_fill(~remaining, -torch.inf)
            index = int(torch.argmax(objective).item())
            chosen.append(index)
            remaining[index] = False
            best_similarity = torch.maximum(best_similarity, similarity[:, index])
        return candidates[
            torch.tensor(chosen, dtype=torch.long, device=candidates.device)
        ]


class LearnedEvidenceSelector(nn.Module):
    """Independent learned saliency and differentiable feature gating (Eq.12)."""

    def __init__(self, *, eta: float = 0.5) -> None:
        super().__init__()
        self.scorer = nn.LazyLinear(1)
        self.eta = float(eta)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        *,
        max_nodes: int,
        strategy: str,
    ) -> BudgetedFeatures:
        scores = (
            torch.sigmoid(self.scorer(features).float())
            .squeeze(-1)
            .masked_fill(~mask, 0.0)
        )
        budgeted = budget_features(
            features=features,
            valid_mask=mask,
            max_nodes=max_nodes,
            strategy=strategy,
            selection_scores=scores,
            eta=self.eta,
        )
        budgeted.ungated_features = budgeted.features
        budgeted.features = budgeted.features * budgeted.selection_scores.to(
            features.dtype
        ).unsqueeze(-1)
        budgeted.features = budgeted.features.masked_fill(
            ~budgeted.mask.unsqueeze(-1), 0.0
        )
        return budgeted
