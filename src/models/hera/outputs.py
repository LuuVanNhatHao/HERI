"""Typed Phase-5 HERA decision output contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.graph.schema import HYPOTHESIS_ORDER


@dataclass
class HERADecisionOutput:
    """Attribution-first HERA decision and grounding outputs."""

    atomic_logits: torch.Tensor
    atomic_probabilities: torch.Tensor
    binary_probability: torch.Tensor
    binary_logit: torch.Tensor
    image_grounding_logits: torch.Tensor
    image_grounding_probabilities: torch.Tensor
    image_grounding_mask: torch.Tensor
    image_grounding_boxes: torch.Tensor
    image_grounding_patch_boxes: torch.Tensor
    text_grounding_logits: torch.Tensor
    text_grounding_probabilities: torch.Tensor
    text_grounding_mask: torch.Tensor
    hypothesis_features: torch.Tensor
    image_source_indices: torch.Tensor | None = None
    text_word_ids: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_support: torch.Tensor | None = None

    def validate(self) -> None:
        """Validate Phase-5 tensor shapes, ranges, masks, and hypothesis order."""

        if self.hypothesis_features.ndim != 3:
            raise ValueError("hypothesis_features must be [B,K,D]")
        batch, hypotheses, _ = self.hypothesis_features.shape
        hypothesis_order = tuple(self.metadata.get("hypothesis_order", HYPOTHESIS_ORDER))
        image_order = tuple(self.metadata.get("image_grounding_order", ("FS", "FA")))
        text_order = tuple(self.metadata.get("text_grounding_order", ("TS", "TA")))
        if hypotheses != len(hypothesis_order):
            raise ValueError("hypothesis_features must preserve configured hypothesis order")
        if self.atomic_logits.shape != (batch, hypotheses):
            raise ValueError(f"atomic_logits must be [B,{hypotheses}]")
        if self.atomic_probabilities.shape != self.atomic_logits.shape:
            raise ValueError("atomic_probabilities shape mismatch")
        if self.binary_probability.shape != (batch,):
            raise ValueError("binary_probability must be [B]")
        if self.binary_logit.shape != (batch,):
            raise ValueError("binary_logit must be [B]")
        if self.image_grounding_logits.ndim != 3 or self.image_grounding_logits.shape[:2] != (batch, len(image_order)):
            raise ValueError("image_grounding_logits must align with image_grounding_order")
        if self.image_grounding_probabilities.shape != self.image_grounding_logits.shape:
            raise ValueError("image_grounding_probabilities shape mismatch")
        if self.image_grounding_mask.shape != (batch, self.image_grounding_logits.shape[-1]):
            raise ValueError("image_grounding_mask must be [B,N_patch]")
        if self.image_grounding_mask.dtype != torch.bool:
            raise ValueError("image_grounding_mask must be bool")
        if self.image_grounding_boxes.shape != (batch, len(image_order), 4):
            raise ValueError("image_grounding_boxes must align with image_grounding_order")
        if self.image_grounding_patch_boxes.shape != (batch, self.image_grounding_logits.shape[-1], 4):
            raise ValueError("image_grounding_patch_boxes must be [B,N_patch,4]")
        if self.text_grounding_logits.ndim != 3 or self.text_grounding_logits.shape[:2] != (batch, len(text_order)):
            raise ValueError("text_grounding_logits must align with text_grounding_order")
        if self.text_grounding_probabilities.shape != self.text_grounding_logits.shape:
            raise ValueError("text_grounding_probabilities shape mismatch")
        if self.text_grounding_mask.shape != (batch, self.text_grounding_logits.shape[-1]):
            raise ValueError("text_grounding_mask must be [B,N_full_text]")
        if self.text_grounding_mask.dtype != torch.bool:
            raise ValueError("text_grounding_mask must be bool")
        for name in (
            "hypothesis_features",
            "atomic_logits",
            "atomic_probabilities",
            "binary_probability",
            "binary_logit",
            "image_grounding_logits",
            "image_grounding_probabilities",
            "image_grounding_boxes",
            "image_grounding_patch_boxes",
            "text_grounding_logits",
            "text_grounding_probabilities",
        ):
            tensor = getattr(self, name)
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{name} contains non-finite values")
        for name in ("atomic_probabilities", "binary_probability", "image_grounding_probabilities", "text_grounding_probabilities"):
            tensor = getattr(self, name)
            if bool(((tensor < -1e-6) | (tensor > 1.0 + 1e-6)).any()):
                raise ValueError(f"{name} must be probabilities in [0,1]")
        for name in ("image_grounding_boxes", "image_grounding_patch_boxes"):
            tensor = getattr(self, name)
            if bool(((tensor < -1e-6) | (tensor > 1.0 + 1e-6)).any()):
                raise ValueError(f"{name} must contain normalized coordinates in [0,1]")
            if bool((tensor[..., 2:] < tensor[..., :2]).any()):
                raise ValueError(f"{name} must contain ordered xyxy boxes")
        if self.image_source_indices is not None and self.image_source_indices.shape != self.image_grounding_mask.shape:
            raise ValueError("image_source_indices must align with image_grounding_mask")
        if self.text_word_ids is not None and self.text_word_ids.shape != self.text_grounding_mask.shape:
            raise ValueError("text_word_ids must align with text_grounding_mask")
        if self.evidence_support is not None:
            if self.evidence_support.shape != (batch, hypotheses):
                raise ValueError("evidence_support must be [B,K]")
            if not torch.isfinite(self.evidence_support).all():
                raise FloatingPointError("evidence_support contains non-finite values")
            if bool(((self.evidence_support < 0) | (self.evidence_support > 1)).any()):
                raise ValueError("evidence_support must be in [0,1]")


@dataclass
class HERADGMOutput(HERADecisionOutput):
    """Full Phase-7 HERA-DGM output with compact integration metadata."""

    evidence_metadata: dict[str, Any] = field(default_factory=dict)
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    graph_metadata: dict[str, Any] = field(default_factory=dict)
    cross_modal_alignment_loss: torch.Tensor | None = None
    router_regularization_loss: torch.Tensor | None = None

    def validate(self) -> None:
        """Validate canonical or explicitly controlled ablation decision contracts."""

        super().validate()
        aggregation = self.metadata.get("binary_aggregation")
        if aggregation == "noisy_or":
            expected_binary = 1.0 - torch.prod(1.0 - self.atomic_probabilities, dim=1)
            if not torch.allclose(self.binary_probability, expected_binary, atol=1e-5, rtol=1e-5):
                raise ValueError("binary_probability must be derived from atomic probabilities by StableNoisyOR")
            if self.metadata.get("binary_shortcut") is not False:
                raise ValueError("Noisy-OR output must record binary_shortcut=False")
        elif aggregation == "independent_head":
            if self.metadata.get("binary_shortcut") is not True:
                raise ValueError("Independent binary ablation must record binary_shortcut=True")
            if not torch.allclose(self.binary_probability, torch.sigmoid(self.binary_logit), atol=1e-6, rtol=1e-6):
                raise ValueError("Independent binary probability must be sigmoid(binary_logit)")
            if self.metadata.get("ablation_variant") != "independent_binary":
                raise ValueError("independent_head is legal only for the independent_binary ablation")
        else:
            raise ValueError(f"Unsupported binary aggregation metadata: {aggregation!r}")
        for name in ("cross_modal_alignment_loss", "router_regularization_loss"):
            value = getattr(self, name)
            if value is not None and (value.ndim != 0 or not torch.isfinite(value)):
                raise ValueError(f"{name} must be a finite scalar")
