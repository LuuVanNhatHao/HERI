"""Generic MTCNN candidate-face grounding head for Phase-2 baselines."""

from __future__ import annotations

import torch
from torch import nn

from src.models.fusion.projections import LazyFeatureProjection
from src.models.outputs import PatchGrid


class ImageGroundingHead(nn.Module):
    """Score legal MTCNN candidate faces using DINO patch features and context."""

    def __init__(self, *, fusion_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.face_projection = LazyFeatureProjection(output_dim=fusion_dim, dropout=dropout)
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
        patch_tokens: torch.Tensor,
        patch_grid: PatchGrid,
        candidate_boxes: torch.Tensor,
        candidate_mask: torch.Tensor,
        fused_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return generic candidate-face grounding logits and mask."""

        face_features = pool_candidate_patches(
            patch_tokens=patch_tokens,
            patch_grid=patch_grid,
            candidate_boxes=candidate_boxes,
            candidate_mask=candidate_mask,
        )
        projected_faces = self.face_projection(face_features)
        projected_context = self.context_projection(fused_context).unsqueeze(1)
        context = projected_context.expand(-1, projected_faces.shape[1], -1)
        logits = self.scorer(torch.cat([projected_faces, context], dim=-1)).squeeze(-1)
        return logits, candidate_mask.to(torch.bool)


def pool_candidate_patches(
    *,
    patch_tokens: torch.Tensor,
    patch_grid: PatchGrid,
    candidate_boxes: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool DINO patch tokens for each candidate box.

    If a tiny face contains no patch center, the nearest patch center to the
    candidate center is selected. This fallback depends only on candidate
    geometry and patch-grid metadata, never on supervision targets.
    """

    if patch_tokens.ndim != 3:
        raise ValueError(f"patch_tokens must be [B,P,C], got {tuple(patch_tokens.shape)}")
    batch, num_patches, channels = patch_tokens.shape
    expected = patch_grid.height * patch_grid.width
    if num_patches != expected:
        raise ValueError(f"Patch token count {num_patches} != grid area {expected}")
    if candidate_boxes.ndim != 3 or candidate_boxes.shape[-1] != 4:
        raise ValueError("candidate_boxes must be [B,F,4]")
    if candidate_boxes.shape[:2] != candidate_mask.shape:
        raise ValueError("candidate_boxes and candidate_mask shape mismatch")

    centers = _patch_centers(
        patch_grid,
        device=patch_tokens.device,
        dtype=patch_tokens.dtype,
    )
    pooled = patch_tokens.new_zeros((batch, candidate_boxes.shape[1], channels * 3))
    for b_idx in range(batch):
        for f_idx in range(candidate_boxes.shape[1]):
            if not bool(candidate_mask[b_idx, f_idx]):
                continue
            box = candidate_boxes[b_idx, f_idx].to(patch_tokens.dtype)
            x1, y1, x2, y2 = box.unbind()
            inside = (
                (centers[:, 0] >= x1)
                & (centers[:, 0] <= x2)
                & (centers[:, 1] >= y1)
                & (centers[:, 1] <= y2)
            )
            if inside.any():
                selected = patch_tokens[b_idx, inside]
                pooled[b_idx, f_idx] = torch.cat(
                    [
                        selected.mean(dim=0),
                        selected.max(dim=0).values,
                        selected.float().std(dim=0, unbiased=False).to(selected.dtype),
                    ],
                    dim=0,
                )
            else:
                center = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
                nearest = torch.argmin(torch.sum((centers - center.unsqueeze(0)) ** 2, dim=1))
                selected = patch_tokens[b_idx, nearest]
                pooled[b_idx, f_idx] = torch.cat([selected, selected, torch.zeros_like(selected)], dim=0)
    return pooled


def _patch_centers(
    patch_grid: PatchGrid,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    xs = (torch.arange(patch_grid.width, device=device, dtype=dtype) + 0.5) * (
        patch_grid.image_width / patch_grid.width
    )
    ys = (torch.arange(patch_grid.height, device=device, dtype=dtype) + 0.5) * (
        patch_grid.image_height / patch_grid.height
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
