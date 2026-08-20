"""Typed Phase-3 evidence output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


PAD_SOURCE_INDEX = -1
PAD_WORD_ID = -1


@dataclass
class EvidenceSet:
    """One graph-budgeted evidence family with explicit provenance."""

    family: str
    features: torch.Tensor
    mask: torch.Tensor
    source_indices: torch.Tensor
    provenance: dict[str, Any] = field(default_factory=dict)
    selection_scores: torch.Tensor | None = None
    spatial_centers: torch.Tensor | None = None
    spatial_boxes: torch.Tensor | None = None
    normalized_centers: torch.Tensor | None = None
    normalized_boxes: torch.Tensor | None = None
    word_ids: torch.Tensor | None = None
    full_features: torch.Tensor | None = None
    full_mask: torch.Tensor | None = None
    full_source_indices: torch.Tensor | None = None
    full_word_ids: torch.Tensor | None = None
    full_spatial_centers: torch.Tensor | None = None
    full_spatial_boxes: torch.Tensor | None = None
    full_normalized_centers: torch.Tensor | None = None
    full_normalized_boxes: torch.Tensor | None = None
    reliability_scores: torch.Tensor | None = None

    def validate(self) -> None:
        """Validate shape, mask, and provenance invariants."""

        if self.features.ndim != 3:
            raise ValueError(f"{self.family}.features must be [B,N,D], got {tuple(self.features.shape)}")
        batch, nodes, _ = self.features.shape
        if self.mask.shape != (batch, nodes):
            raise ValueError(f"{self.family}.mask shape mismatch: {tuple(self.mask.shape)}")
        if self.source_indices.shape != (batch, nodes):
            raise ValueError(f"{self.family}.source_indices shape mismatch")
        if self.mask.dtype != torch.bool:
            raise ValueError(f"{self.family}.mask must be bool")
        if self.source_indices.dtype != torch.long:
            raise ValueError(f"{self.family}.source_indices must be torch.long")
        if not torch.isfinite(self.features).all():
            raise FloatingPointError(f"{self.family}.features contains non-finite values")
        padded_sources = self.source_indices[~self.mask]
        if padded_sources.numel() and not torch.all(padded_sources == PAD_SOURCE_INDEX):
            raise ValueError(f"{self.family}.source_indices must be -1 where mask is false")
        self._validate_optional_tensor(self.selection_scores, "selection_scores", (batch, nodes))
        self._validate_optional_tensor(self.word_ids, "word_ids", (batch, nodes))
        self._validate_optional_tensor(self.spatial_centers, "spatial_centers", (batch, nodes, 2))
        self._validate_optional_tensor(self.spatial_boxes, "spatial_boxes", (batch, nodes, 4))
        self._validate_optional_tensor(self.normalized_centers, "normalized_centers", (batch, nodes, 2))
        self._validate_optional_tensor(self.normalized_boxes, "normalized_boxes", (batch, nodes, 4))
        self._validate_optional_tensor(self.reliability_scores, "reliability_scores", (batch, nodes))
        if self.reliability_scores is not None:
            valid_scores = self.reliability_scores[self.mask]
            if valid_scores.numel() and (
                (valid_scores < -1e-6).any() or (valid_scores > 1.0 + 1e-6).any()
            ):
                raise ValueError(f"{self.family}.reliability_scores must stay in [0,1]")
            if bool((self.reliability_scores[~self.mask] != 0).any()):
                raise ValueError(f"{self.family}.reliability_scores must be zero for padded nodes")
        self._validate_coordinates()
        self._validate_full_store()

    def valid_counts(self) -> torch.Tensor:
        """Return valid evidence counts per sample."""

        return self.mask.sum(dim=1)

    def _validate_optional_tensor(
        self,
        tensor: torch.Tensor | None,
        name: str,
        expected_shape: tuple[int, ...],
    ) -> None:
        if tensor is None:
            return
        if tensor.shape != expected_shape:
            raise ValueError(f"{self.family}.{name} must be {expected_shape}, got {tuple(tensor.shape)}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{self.family}.{name} contains non-finite values")

    def _validate_coordinates(self) -> None:
        for name in ("normalized_centers", "normalized_boxes"):
            tensor = getattr(self, name)
            if tensor is None or not bool(self.mask.any()):
                continue
            values = tensor[self.mask]
            if values.numel() and ((values < -1e-6).any() or (values > 1.0 + 1e-6).any()):
                raise ValueError(f"{self.family}.{name} must stay in [0,1] for valid nodes")
        for name in ("spatial_centers", "spatial_boxes"):
            tensor = getattr(self, name)
            if tensor is None or not bool(self.mask.any()):
                continue
            values = tensor[self.mask]
            if values.numel() and (values < -1e-6).any():
                raise ValueError(f"{self.family}.{name} cannot contain negative coordinates")

    def _validate_full_store(self) -> None:
        if self.full_features is None:
            return
        if self.full_features.ndim != 3:
            raise ValueError(f"{self.family}.full_features must be [B,N,C]")
        batch, full_nodes, _ = self.full_features.shape
        if batch != self.features.shape[0]:
            raise ValueError(f"{self.family}.full_features batch mismatch")
        if self.full_mask is not None and self.full_mask.shape != (batch, full_nodes):
            raise ValueError(f"{self.family}.full_mask shape mismatch")
        if self.full_source_indices is not None and self.full_source_indices.shape != (batch, full_nodes):
            raise ValueError(f"{self.family}.full_source_indices shape mismatch")
        if self.full_word_ids is not None and self.full_word_ids.shape != (batch, full_nodes):
            raise ValueError(f"{self.family}.full_word_ids shape mismatch")
        for name, width in (
            ("full_spatial_centers", 2),
            ("full_spatial_boxes", 4),
            ("full_normalized_centers", 2),
            ("full_normalized_boxes", 4),
        ):
            value = getattr(self, name)
            if value is not None:
                expected = (batch, full_nodes, width)
                if value.shape != expected:
                    raise ValueError(f"{self.family}.{name} must be {expected}")
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"{self.family}.{name} contains non-finite values")
        for name in ("full_normalized_centers", "full_normalized_boxes"):
            value = getattr(self, name)
            if value is not None and value.numel() and (
                (value < -1e-6).any() or (value > 1.0 + 1e-6).any()
            ):
                raise ValueError(f"{self.family}.{name} must stay in [0,1]")
        if not bool(self.mask.any()):
            return
        selected = self.source_indices[self.mask]
        if (selected < 0).any() or (selected >= full_nodes).any():
            raise ValueError(f"{self.family}.source_indices out of range for full store")
        if self.full_mask is not None:
            gathered = self.full_mask.gather(1, self.source_indices.clamp_min(0))
            if not torch.all(gathered[self.mask]):
                raise ValueError(f"{self.family}.source_indices refer to masked full-store entries")


@dataclass
class CrossModalMetadata:
    """Target-independent semantic similarity metadata."""

    similarity: torch.Tensor | None = None
    visual_mask: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    visual_source_indices: torch.Tensor | None = None
    text_source_indices: torch.Tensor | None = None
    edge_confidence: torch.Tensor | None = None
    alignment_loss: torch.Tensor | None = None
    method: str = "disabled"
    feature_space: str = "disabled"

    def validate(self) -> None:
        """Validate similarity and mask alignment."""

        if self.similarity is None:
            return
        if self.similarity.ndim != 3:
            raise ValueError(f"cross_modal.similarity must be [B,V,T], got {tuple(self.similarity.shape)}")
        batch, visual_nodes, text_nodes = self.similarity.shape
        if self.visual_mask is None or self.visual_mask.shape != (batch, visual_nodes):
            raise ValueError("cross_modal.visual_mask shape mismatch")
        if self.text_mask is None or self.text_mask.shape != (batch, text_nodes):
            raise ValueError("cross_modal.text_mask shape mismatch")
        if not torch.isfinite(self.similarity).all():
            raise FloatingPointError("cross_modal.similarity contains non-finite values")
        valid = self.visual_mask.unsqueeze(2) & self.text_mask.unsqueeze(1)
        if bool((self.similarity[~valid] != 0).any()):
            raise ValueError("cross_modal.similarity must be zero where either side is masked")
        if self.edge_confidence is not None:
            if self.edge_confidence.shape != self.similarity.shape:
                raise ValueError("cross_modal.edge_confidence shape mismatch")
            if not torch.isfinite(self.edge_confidence).all():
                raise FloatingPointError("cross_modal.edge_confidence contains non-finite values")
            if bool(((self.edge_confidence < 0) | (self.edge_confidence > 1)).any()):
                raise ValueError("cross_modal.edge_confidence must be in [0,1]")
            if bool((self.edge_confidence[~valid] != 0).any()):
                raise ValueError("cross_modal.edge_confidence must be zero where either side is masked")
        if self.alignment_loss is not None:
            if self.alignment_loss.ndim != 0 or not torch.isfinite(self.alignment_loss):
                raise ValueError("cross_modal.alignment_loss must be a finite scalar")


@dataclass
class HeterogeneousEvidenceBatch:
    """All Phase-3 evidence families produced for one batch."""

    visual_semantic: EvidenceSet
    visual_forensic: EvidenceSet
    face: EvidenceSet
    text_alignment: EvidenceSet
    text_context: EvidenceSet
    cross_modal_metadata: CrossModalMetadata = field(default_factory=CrossModalMetadata)

    def families(self) -> dict[str, EvidenceSet]:
        """Return evidence families by canonical name."""

        return {
            "visual_semantic": self.visual_semantic,
            "visual_forensic": self.visual_forensic,
            "face": self.face,
            "text_alignment": self.text_alignment,
            "text_context": self.text_context,
        }

    def validate(self) -> None:
        """Validate every family and cross-modal metadata."""

        for evidence_set in self.families().values():
            evidence_set.validate()
        self.cross_modal_metadata.validate()
