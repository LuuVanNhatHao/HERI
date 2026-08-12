"""Diagnostics for Phase-6 evidence reliability routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .outputs import RoutedEvidenceBatch, RoutedEvidenceSet


@dataclass(frozen=True)
class RoutingDiagnostics:
    """Diagnostic thresholds for routing inspection only."""

    low_threshold: float
    high_threshold: float


def summarize_routed_batch(
    routed: RoutedEvidenceBatch,
    *,
    diagnostics: RoutingDiagnostics,
) -> dict[str, Any]:
    """Return concise JSON-friendly per-family reliability diagnostics."""

    routed.validate()
    return {
        "router": dict(routed.metadata),
        "families": {
            name: summarize_routed_set(routed_set, diagnostics=diagnostics)
            for name, routed_set in routed.families().items()
        },
    }


def summarize_routed_set(
    routed_set: RoutedEvidenceSet,
    *,
    diagnostics: RoutingDiagnostics,
) -> dict[str, Any]:
    """Summarize one routed family without printing raw scores."""

    routed_set.validate()
    valid_scores = routed_set.reliability_scores[routed_set.mask]
    if valid_scores.numel():
        mean = float(valid_scores.detach().mean().cpu())
        std = float(valid_scores.detach().std(unbiased=False).cpu())
        minimum = float(valid_scores.detach().min().cpu())
        maximum = float(valid_scores.detach().max().cpu())
        low_fraction = float((valid_scores < diagnostics.low_threshold).float().mean().detach().cpu())
        high_fraction = float((valid_scores > diagnostics.high_threshold).float().mean().detach().cpu())
    else:
        mean = std = minimum = maximum = low_fraction = high_fraction = 0.0
    return {
        "feature_shape": list(routed_set.routed_features.shape),
        "valid_nodes_per_sample": [int(value) for value in routed_set.mask.sum(dim=1).detach().cpu().tolist()],
        "effective_nodes_per_sample": [
            int(value) for value in (routed_set.effective_mask if routed_set.effective_mask is not None else routed_set.mask).sum(dim=1).detach().cpu().tolist()
        ],
        "reliability_mean": mean,
        "reliability_std": std,
        "reliability_min": minimum,
        "reliability_max": maximum,
        "fraction_below_low_threshold": low_fraction,
        "fraction_above_high_threshold": high_fraction,
        "source_indices_shape": list(routed_set.source_indices.shape),
    }
