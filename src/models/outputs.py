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
    text_grounding_logits: torch.Tensor
    text_grounding_mask: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate essential tensor shape relationships."""

        batch = self.binary_logit.shape[0]
        if self.binary_logit.shape != (batch,):
            raise ValueError(f"binary_logit must be [B], got {tuple(self.binary_logit.shape)}")
        if self.atomic_logits.shape != (batch, 4):
            raise ValueError(f"atomic_logits must be [B,4], got {tuple(self.atomic_logits.shape)}")
        if self.image_grounding_logits.shape != self.image_grounding_mask.shape:
            raise ValueError("image grounding logits/mask shape mismatch")
        if self.text_grounding_logits.shape != self.text_grounding_mask.shape:
            raise ValueError("text grounding logits/mask shape mismatch")
