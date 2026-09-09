"""B1 strong cross-attention baseline."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.models.fusion.cross_attention import CrossAttentionFusion
from src.models.heads.attribution import AttributionHead
from src.models.heads.binary import BinaryHead
from src.models.heads.image_grounding import ImageGroundingHead
from src.models.heads.text_grounding import TextGroundingHead
from src.models.heads.noisy_or import StableNoisyOR
from src.models.outputs import BackboneOutputs, BaselineOutput

from .base import BaselineBackboneMixin, finite_or_fail


class B1CrossAttentionBaseline(nn.Module, BaselineBackboneMixin):
    """Conventional token/patch cross-attention baseline using the same backbones."""

    def __init__(
        self,
        *,
        fusion_dim: int,
        num_layers: int,
        num_heads: int,
        head_hidden_dim: int,
        dropout: float,
        binary_aggregation: str = "independent_head",
        hypothesis_order: tuple[str, ...] = ("FS", "FA", "TS", "TA"),
        image_hypothesis_names: tuple[str, ...] = ("FS", "FA"),
        siglip2: nn.Module | None = None,
        dinov3: nn.Module | None = None,
        deberta: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.siglip2 = siglip2
        self.dinov3 = dinov3
        self.deberta = deberta
        self.hypothesis_order = tuple(hypothesis_order)
        self.image_hypothesis_names = tuple(image_hypothesis_names)
        if not self.image_hypothesis_names or not set(self.image_hypothesis_names).issubset(
            self.hypothesis_order
        ):
            raise ValueError("image_hypothesis_names must be a non-empty ontology subset")
        self.image_hypothesis_indices = tuple(
            self.hypothesis_order.index(name) for name in self.image_hypothesis_names
        )
        self.fusion = CrossAttentionFusion(
            fusion_dim=fusion_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        if binary_aggregation not in {"independent_head", "noisy_or"}:
            raise ValueError("B1 binary aggregation must be independent_head or noisy_or")
        self.binary_aggregation = binary_aggregation
        if binary_aggregation == "independent_head":
            self.binary_head = BinaryHead(input_dim=fusion_dim)
        self.noisy_or = StableNoisyOR()
        self.attribution_head = AttributionHead(
            input_dim=fusion_dim, num_labels=len(self.hypothesis_order)
        )
        self.image_grounding_head = ImageGroundingHead(
            fusion_dim=fusion_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        self.text_grounding_head = TextGroundingHead(
            fusion_dim=fusion_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        features: BackboneOutputs | None = None,
    ) -> BaselineOutput:
        encoded = self._encode_backbones(batch, features)
        fused, fused_tokens, fused_mask = self.fusion(
            siglip_image_tokens=encoded.siglip2.image_tokens,
            siglip_image_mask=encoded.siglip2.image_mask,
            siglip_text_tokens=encoded.siglip2.text_tokens,
            siglip_text_mask=encoded.siglip2.text_mask,
            dino_tokens=encoded.dinov3.patch_tokens,
            dino_mask=encoded.dinov3.patch_mask,
            deberta_tokens=encoded.deberta.token_features,
            deberta_mask=encoded.deberta.attention_mask,
        )
        atomic_logits = self.attribution_head(fused)
        binary_logit = (
            self.noisy_or(atomic_logits).binary_logit
            if self.binary_aggregation == "noisy_or"
            else self.binary_head(fused)
        )
        image_logits, image_mask, image_boxes, image_patch_boxes = self.image_grounding_head(
            patch_tokens=encoded.dinov3.patch_tokens,
            patch_mask=encoded.dinov3.patch_mask,
            patch_grid=encoded.dinov3.patch_grid,
            fused_context=fused,
        )
        text_logits, text_mask = self.text_grounding_head(
            token_features=encoded.deberta.token_features,
            token_mask=batch["text_grounding_mask"],
            fused_context=fused,
        )
        finite_or_fail(
            {
                "binary_logit": binary_logit,
                "atomic_logits": atomic_logits,
                "image_grounding_logits": image_logits,
                "text_grounding_logits": text_logits,
            }
        )
        output = BaselineOutput(
            binary_logit=binary_logit,
            binary_probability=torch.sigmoid(binary_logit),
            atomic_logits=atomic_logits,
            atomic_probabilities=torch.sigmoid(atomic_logits),
            image_grounding_logits=image_logits,
            image_grounding_mask=image_mask,
            image_grounding_boxes=image_boxes,
            image_grounding_patch_boxes=image_patch_boxes,
            text_grounding_logits=text_logits,
            text_grounding_mask=text_mask,
            metadata={
                "baseline_type": "b1_cross_attention",
                "hypothesis_order": list(self.hypothesis_order),
                "image_grounding_order": list(self.image_hypothesis_names),
                "image_hypothesis_indices": list(self.image_hypothesis_indices),
                "image_presence_aggregation": "ontology_noisy_or",
                "binary_aggregation": self.binary_aggregation,
                "fused_token_count": fused_tokens.shape[1],
                "fused_valid_tokens": fused_mask.sum(dim=1).detach().cpu().tolist(),
            },
        )
        output.validate()
        return output
