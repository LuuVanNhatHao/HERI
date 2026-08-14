"""Typed Phase-6 routed evidence output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from src.evidence.outputs import CrossModalMetadata, EvidenceSet, HeterogeneousEvidenceBatch, PAD_SOURCE_INDEX


@dataclass
class RoutedEvidenceSet:
    """One evidence family after target-independent reliability routing."""

    family: str
    original_features: torch.Tensor
    routed_features: torch.Tensor
    mask: torch.Tensor
    reliability_logits: torch.Tensor
    reliability_scores: torch.Tensor
    source_indices: torch.Tensor
    provenance: dict[str, Any] = field(default_factory=dict)
    original: EvidenceSet | None = None
    effective_mask: torch.Tensor | None = None

    def validate(self) -> None:
        """Validate routed tensor shapes, finite values, score ranges, and masks."""

        if self.original_features.ndim != 3:
            raise ValueError(f"{self.family}.original_features must be [B,N,D]")
        if self.routed_features.shape != self.original_features.shape:
            raise ValueError(f"{self.family}.routed_features shape mismatch")
        batch, nodes, _ = self.original_features.shape
        if self.mask.shape != (batch, nodes):
            raise ValueError(f"{self.family}.mask shape mismatch")
        if self.mask.dtype != torch.bool:
            raise ValueError(f"{self.family}.mask must be bool")
        if self.reliability_logits.shape != (batch, nodes):
            raise ValueError(f"{self.family}.reliability_logits must be [B,N]")
        if self.reliability_scores.shape != (batch, nodes):
            raise ValueError(f"{self.family}.reliability_scores must be [B,N]")
        if self.source_indices.shape != (batch, nodes):
            raise ValueError(f"{self.family}.source_indices shape mismatch")
        if self.effective_mask is not None:
            if self.effective_mask.shape != self.mask.shape:
                raise ValueError(f"{self.family}.effective_mask shape mismatch")
            if self.effective_mask.dtype != torch.bool:
                raise ValueError(f"{self.family}.effective_mask must be bool")
            if bool((self.effective_mask & ~self.mask).any()):
                raise ValueError(f"{self.family}.effective_mask cannot activate padded nodes")
        for name in ("original_features", "routed_features", "reliability_logits", "reliability_scores"):
            tensor = getattr(self, name)
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{self.family}.{name} contains non-finite values")
        if bool(((self.reliability_scores < -1e-6) | (self.reliability_scores > 1.0 + 1e-6)).any()):
            raise ValueError(f"{self.family}.reliability_scores must be in [0,1]")
        if bool((self.reliability_scores[~self.mask] != 0).any()):
            raise ValueError(f"{self.family}.reliability_scores must be zero for padded nodes")
        if bool((self.routed_features[~self.mask] != 0).any()):
            raise ValueError(f"{self.family}.routed_features must be zero for padded nodes")

    def as_evidence_set(self) -> EvidenceSet:
        """Return graph-compatible EvidenceSet with routed features and preserved metadata."""

        if self.original is None:
            raise ValueError(f"{self.family}.original evidence reference is required")
        graph_mask = self.mask if self.effective_mask is None else self.effective_mask
        source_indices = self.original.source_indices
        if graph_mask is not self.original.mask:
            source_indices = source_indices.masked_fill(~graph_mask, PAD_SOURCE_INDEX)
        return replace(
            self.original,
            features=self.routed_features,
            mask=graph_mask,
            source_indices=source_indices,
        )


@dataclass
class RoutedEvidenceBatch:
    """All five Phase-6 routed evidence families plus preserved cross-modal metadata."""

    visual_semantic: RoutedEvidenceSet
    visual_forensic: RoutedEvidenceSet
    face: RoutedEvidenceSet
    text_alignment: RoutedEvidenceSet
    text_context: RoutedEvidenceSet
    cross_modal_metadata: CrossModalMetadata
    original: HeterogeneousEvidenceBatch
    metadata: dict[str, Any] = field(default_factory=dict)
    regularization_loss: torch.Tensor | None = None

    def families(self) -> dict[str, RoutedEvidenceSet]:
        """Return routed families by canonical name."""

        return {
            "visual_semantic": self.visual_semantic,
            "visual_forensic": self.visual_forensic,
            "face": self.face,
            "text_alignment": self.text_alignment,
            "text_context": self.text_context,
        }

    def validate(self) -> None:
        """Validate every routed family and graph-compatible conversion."""

        self.original.validate()
        for name, routed_set in self.families().items():
            routed_set.validate()
            original_set = self.original.families()[name]
            if routed_set.original is not original_set:
                raise ValueError(f"{name}.original must preserve the Phase-3 EvidenceSet reference")
            if routed_set.provenance != original_set.provenance:
                raise ValueError(f"{name}.provenance must be preserved unchanged")
        self.cross_modal_metadata.validate()
        if self.regularization_loss is not None:
            if self.regularization_loss.ndim != 0 or not torch.isfinite(self.regularization_loss):
                raise ValueError("routing regularization_loss must be a finite scalar")
        self.as_graph_evidence().validate()

    def as_graph_evidence(self) -> HeterogeneousEvidenceBatch:
        """Return a Phase-4-compatible batch using routed graph-budgeted features."""

        return HeterogeneousEvidenceBatch(
            visual_semantic=self.visual_semantic.as_evidence_set(),
            visual_forensic=self.visual_forensic.as_evidence_set(),
            face=self.face.as_evidence_set(),
            text_alignment=self.text_alignment.as_evidence_set(),
            text_context=self.text_context.as_evidence_set(),
            cross_modal_metadata=self.cross_modal_metadata,
        )
