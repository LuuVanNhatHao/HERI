"""B1 conventional multimodal cross-attention fusion."""

from __future__ import annotations

import torch
from torch import nn

from src.models.encoders.common import masked_mean

from .projections import LazyFeatureProjection


class CrossAttentionFusion(nn.Module):
    """Project token streams and fuse them with TransformerEncoder attention."""

    def __init__(
        self,
        *,
        fusion_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.siglip_image = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        self.siglip_text = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        self.dino = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        self.deberta = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        *,
        siglip_image_tokens: torch.Tensor,
        siglip_image_mask: torch.Tensor | None,
        siglip_text_tokens: torch.Tensor,
        siglip_text_mask: torch.Tensor,
        dino_tokens: torch.Tensor,
        dino_mask: torch.Tensor,
        deberta_tokens: torch.Tensor,
        deberta_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused global representation, fused tokens, and token mask."""

        streams = [
            self.siglip_image(siglip_image_tokens),
            self.siglip_text(siglip_text_tokens),
            self.dino(dino_tokens),
            self.deberta(deberta_tokens),
        ]
        masks = [
            siglip_image_mask
            if siglip_image_mask is not None
            else torch.ones(siglip_image_tokens.shape[:2], dtype=torch.bool, device=siglip_image_tokens.device),
            siglip_text_mask,
            dino_mask,
            deberta_mask,
        ]
        tokens = torch.cat(streams, dim=1)
        mask = torch.cat([m.to(torch.bool) for m in masks], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        encoded = self.final_norm(encoded)
        return masked_mean(encoded, mask), encoded, mask
