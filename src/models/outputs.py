"""Typed model output contracts for Phase 2 baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PatchGrid:
    """Spatial metadata for a sequence of patch tokens."""

    height: int
    width: int
    image_height: int
    image_width: int


@dataclass
class SigLIP2Output:
    """Semantic multimodal encoder output."""

    image_tokens: torch.Tensor
    image_mask: torch.Tensor | None
    image_global: torch.Tensor
    text_tokens: torch.Tensor
    text_mask: torch.Tensor
    text_global: torch.Tensor
    image_grid: PatchGrid | None = None
    image_special_token_count: int = 0
    text_word_ids: torch.Tensor | None = None


@dataclass
class DINOv3Output:
    """Dense visual encoder output."""

    patch_tokens: torch.Tensor
    patch_mask: torch.Tensor
    global_feature: torch.Tensor
    patch_grid: PatchGrid
    patch_special_token_count: int = 0


@dataclass
class DeBERTaOutput:
    """Contextual text encoder output."""

    token_features: torch.Tensor
    attention_mask: torch.Tensor
    global_feature: torch.Tensor


@dataclass
class BackboneOutputs:
    """All three Phase-2 backbone outputs."""

    siglip2: SigLIP2Output
    dinov3: DINOv3Output
    deberta: DeBERTaOutput


@dataclass
class BaselineOutput:
    """Structured B0/B1 baseline output contract."""

    binary_logit: torch.Tensor
    binary_probability: torch.Tensor
    atomic_logits: torch.Tensor
    atomic_probabilities: torch.Tensor
    image_grounding_logits: torch.Tensor
    image_grounding_mask: torch.Tensor
    image_grounding_boxes: torch.Tensor
    image_grounding_patch_boxes: torch.Tensor
    text_grounding_logits: torch.Tensor
    text_grounding_mask: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate essential tensor shape relationships."""

        batch = self.binary_logit.shape[0]
        if self.binary_logit.shape != (batch,):
            raise ValueError(f"binary_logit must be [B], got {tuple(self.binary_logit.shape)}")
        expected = self.metadata.get("hypothesis_order")
        hypotheses = len(expected) if expected else self.atomic_logits.shape[1]
        if self.atomic_logits.shape != (batch, hypotheses) or hypotheses < 1:
            raise ValueError(
                f"atomic_logits must be [B,{hypotheses}], got {tuple(self.atomic_logits.shape)}"
            )
        if self.atomic_probabilities.shape != self.atomic_logits.shape:
            raise ValueError("atomic_probabilities must align with atomic_logits")
        if self.image_grounding_logits.shape != self.image_grounding_mask.shape:
            raise ValueError("image grounding logits/mask shape mismatch")
        if self.image_grounding_boxes.shape != (batch, 4):
            raise ValueError("image_grounding_boxes must be [B,4]")
        if self.image_grounding_patch_boxes.shape != (batch, self.image_grounding_logits.shape[-1], 4):
            raise ValueError("image_grounding_patch_boxes must be [B,N_patch,4]")
        for name in ("image_grounding_boxes", "image_grounding_patch_boxes"):
            boxes = getattr(self, name)
            if not torch.isfinite(boxes).all():
                raise FloatingPointError(f"{name} contains non-finite values")
            if bool(((boxes < -1e-6) | (boxes > 1.0 + 1e-6)).any()):
                raise ValueError(f"{name} must be normalized to [0,1]")
            if bool((boxes[..., 2:] < boxes[..., :2]).any()):
                raise ValueError(f"{name} must contain ordered xyxy boxes")
        if self.text_grounding_logits.shape != self.text_grounding_mask.shape:
            raise ValueError("text grounding logits/mask shape mismatch")
