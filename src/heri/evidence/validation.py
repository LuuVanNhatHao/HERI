"""Evidence validation and inspection utilities."""

from __future__ import annotations

from typing import Any

import torch

from .outputs import EvidenceSet, HeterogeneousEvidenceBatch


def validate_evidence_batch(evidence: HeterogeneousEvidenceBatch) -> None:
    """Fail fast on evidence contract violations."""

    evidence.validate()


def summarize_evidence_batch(evidence: HeterogeneousEvidenceBatch) -> dict[str, Any]:
    """Return concise JSON-friendly diagnostics for inspection scripts."""

    summary: dict[str, Any] = {}
    for name, evidence_set in evidence.families().items():
        summary[name] = summarize_evidence_set(evidence_set)
    cross = evidence.cross_modal_metadata
    summary["cross_modal"] = {
        "method": cross.method,
        "similarity_shape": list(cross.similarity.shape) if cross.similarity is not None else None,
    }
    return summary


def summarize_evidence_set(evidence_set: EvidenceSet) -> dict[str, Any]:
    """Return shape, counts, finite-rate, and norm diagnostics."""

    features = evidence_set.features.detach()
    mask = evidence_set.mask.detach()
    finite = torch.isfinite(features)
    valid_features = features[mask] if bool(mask.any()) else features.new_zeros((0, features.shape[-1]))
    norms = torch.linalg.vector_norm(valid_features, dim=-1) if valid_features.numel() else features.new_zeros((0,))
    return {
        "feature_shape": list(features.shape),
        "mask_shape": list(mask.shape),
        "valid_nodes_per_sample": [int(value) for value in mask.sum(dim=1).detach().cpu().tolist()],
        "source_indices_shape": list(evidence_set.source_indices.shape),
        "finite_rate": float(finite.float().mean().item()) if features.numel() else 1.0,
        "norm_mean": float(norms.mean().item()) if norms.numel() else 0.0,
        "norm_max": float(norms.max().item()) if norms.numel() else 0.0,
        "spatial_boxes_shape": list(evidence_set.spatial_boxes.shape) if evidence_set.spatial_boxes is not None else None,
        "word_ids_shape": list(evidence_set.word_ids.shape) if evidence_set.word_ids is not None else None,
    }
