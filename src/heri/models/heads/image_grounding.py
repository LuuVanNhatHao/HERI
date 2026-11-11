"""Dense raw-image grounding heads."""

from __future__ import annotations

import torch
from torch import nn

from heri.models.fusion.projections import LazyFeatureProjection
from heri.models.outputs import PatchGrid
from heri.evidence.spatial import patch_centers_and_boxes


class ImageGroundingHead(nn.Module):
    """Predict a manipulation heatmap and normalized bbox from DINO patches."""

    def __init__(self, *, fusion_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.patch_projection = nn.LazyLinear(fusion_dim, bias=False)
        self.context_projection = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.grounding_bias = nn.Parameter(torch.zeros(()))
        self.box_regressor = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.GELU(),
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
        logits = (projected_patches.float() * projected_context.float()).sum(dim=-1) + self.grounding_bias.float()
        logits = logits.masked_fill(~mask, -1e4)
        from .hypothesis_grounding import _masked_softmax
        attention = _masked_softmax(logits, mask)
        localized_context = torch.sum(attention.unsqueeze(-1) * projected_patches.float(), dim=1).to(fused_context.dtype)
        _, _, _, normalized_boxes = patch_centers_and_boxes(
            patch_grid,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        patch_boxes = normalized_boxes.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        box = _normalized_center_size_box(
            self.box_regressor(
                torch.cat([fused_context, localized_context], dim=-1)
            )
        )
        return logits, mask, box, patch_boxes


def _normalized_center_size_box(parameters: torch.Tensor) -> torch.Tensor:
    if parameters.ndim != 2 or parameters.shape[-1] != 4:
        raise ValueError("bbox parameters must be [B,4]")
    center_size = torch.sigmoid(parameters.float())
    center, size = center_size[:, :2], center_size[:, 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)
