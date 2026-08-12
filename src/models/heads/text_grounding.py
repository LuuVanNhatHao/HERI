"""Generic manipulated-token grounding head for Phase-2 baselines."""

from __future__ import annotations

import torch
from torch import nn

from src.models.fusion.projections import LazyFeatureProjection


class TextGroundingHead(nn.Module):
    """Score DeBERTa subwords using token features conditioned on fused context."""

    def __init__(self, *, fusion_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.token_projection = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        token_features: torch.Tensor,
        token_mask: torch.Tensor,
        fused_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token logits and the unmodified valid-token mask."""

        projected_tokens = self.token_projection(token_features)
        projected_context = self.context_projection(fused_context).unsqueeze(1)
        context = projected_context.expand(-1, projected_tokens.shape[1], -1)
        logits = self.scorer(torch.cat([projected_tokens, context], dim=-1)).squeeze(-1)
        return logits, token_mask.to(torch.bool)
