"""Dense raw-image grounding heads."""

from __future__ import annotations

import torch
from torch import nn

from src.models.fusion.projections import LazyFeatureProjection
from src.models.outputs import PatchGrid
from src.evidence.spatial import patch_centers_and_boxes


class ImageGroundingHead(nn.Module):
    """Predict a manipulation heatmap and normalized bbox from DINO patches."""

    def __init__(self, *, fusion_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.patch_projection = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
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
        self.box_regressor = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2 + 4),
            nn.Linear(fusion_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        *,
        patch_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_grid: PatchGrid,
        fused_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return patch logits/mask, predicted box, and normalized patch boxes."""

        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must be [B,P,C]")
        if patch_mask.shape != patch_tokens.shape[:2]:
            raise ValueError("patch_mask must align with patch_tokens")
        mask = patch_mask.to(torch.bool)
        projected_patches = self.patch_projection(patch_tokens)
        projected_context = self.context_projection(fused_context).unsqueeze(1)
        context = projected_context.expand(-1, projected_patches.shape[1], -1)
        logits = self.scorer(torch.cat([projected_patches, context], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        attention = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        localized_context = torch.sum(attention.unsqueeze(-1) * projected_patches, dim=1)
        _, _, _, normalized_boxes = patch_centers_and_boxes(
            patch_grid,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        patch_boxes = normalized_boxes.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        geometric_prior = torch.sum(attention.unsqueeze(-1) * patch_boxes, dim=1)
        box = _normalized_center_size_box(
            self.box_regressor(
                torch.cat([fused_context, localized_context, geometric_prior], dim=-1)
            )
        )
        return logits, mask, box, patch_boxes


def _normalized_center_size_box(parameters: torch.Tensor) -> torch.Tensor:
    if parameters.ndim != 2 or parameters.shape[-1] != 4:
        raise ValueError("bbox parameters must be [B,4]")
    anchor = torch.sigmoid(parameters[:, :2])
    size = torch.sigmoid(parameters[:, 2:]).clamp(1e-4, 1.0)
    top_left = anchor * (1.0 - size)
    return torch.cat([top_left, top_left + size], dim=-1)
