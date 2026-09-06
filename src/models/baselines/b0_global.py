"""B0 strong global fusion baseline."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.models.fusion.global_fusion import GlobalFusion
from src.models.heads.attribution import AttributionHead
from src.models.heads.binary import BinaryHead
from src.models.heads.image_grounding import ImageGroundingHead
from src.models.heads.text_grounding import TextGroundingHead
from src.models.outputs import BackboneOutputs, BaselineOutput

from .base import BaselineBackboneMixin, finite_or_fail


class B0GlobalFusionBaseline(nn.Module, BaselineBackboneMixin):
    """SigLIP2 + DINOv3 + DeBERTa conventional global fusion baseline."""

    def __init__(
        self,
        *,
        fusion_dim: int,
        fusion_hidden_dim: int,
        head_hidden_dim: int,
        dropout: float,
        hypothesis_order: tuple[str, ...] = ("FS", "FA", "TS", "TA"),
        siglip2: nn.Module | None = None,
        dinov3: nn.Module | None = None,
        deberta: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.siglip2 = siglip2
        self.dinov3 = dinov3
        self.deberta = deberta
        self.fusion = GlobalFusion(
            fusion_dim=fusion_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_dim,
            dropout=dropout,
        )
        self.binary_head = BinaryHead(input_dim=fusion_dim)
        self.hypothesis_order = tuple(hypothesis_order)
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
        fused = self.fusion(
            siglip_image_global=encoded.siglip2.image_global,
            siglip_text_global=encoded.siglip2.text_global,
            dino_global=encoded.dinov3.global_feature,
            deberta_global=encoded.deberta.global_feature,
        )
        binary_logit = self.binary_head(fused)
        atomic_logits = self.attribution_head(fused)
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
            metadata={"baseline_type": "b0_global", "hypothesis_order": list(self.hypothesis_order)},
        )
        output.validate()
        return output
