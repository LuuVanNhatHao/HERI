"""B0 conventional global feature fusion."""

from __future__ import annotations

import torch
from torch import nn

from .projections import LazyFeatureProjection, MLPBlock


class GlobalFusion(nn.Module):
    """Project four global streams and fuse by concatenation + MLP."""

    def __init__(
        self,
        *,
        fusion_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.siglip_image = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        self.siglip_text = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        self.dino = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        self.deberta = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        self.fusion = MLPBlock(
            input_dim=fusion_dim * 4,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        *,
        siglip_image_global: torch.Tensor,
        siglip_text_global: torch.Tensor,
        dino_global: torch.Tensor,
        deberta_global: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fused B0 representation."""

        projected = [
            self.siglip_image(siglip_image_global),
            self.siglip_text(siglip_text_global),
            self.dino(dino_global),
            self.deberta(deberta_global),
        ]
        return self.fusion(torch.cat(projected, dim=-1))
