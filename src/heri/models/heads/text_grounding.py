"""Generic manipulated-token grounding head for baselines."""

from __future__ import annotations

import torch
from torch import nn

from heri.models.fusion.projections import LazyFeatureProjection


class TextGroundingHead(nn.Module):
    """Score DeBERTa subwords using token features conditioned on fused context."""

    def __init__(self, *, fusion_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.token_projection = nn.LazyLinear(fusion_dim, bias=False)
        self.context_projection = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.grounding_bias = nn.Parameter(torch.zeros(()))

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
        logits = (projected_tokens.float() * projected_context.float()).sum(dim=-1) + self.grounding_bias.float()
        return logits, token_mask.to(torch.bool)
