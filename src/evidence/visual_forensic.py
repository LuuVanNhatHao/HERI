"""DINOv3 dense visual-forensic evidence extraction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.outputs import DINOv3Output

from .budgeting import budget_features
from .outputs import EvidenceSet
from .spatial import gather_spatial, patch_centers_and_boxes, validate_patch_grid


class VisualForensicEvidenceExtractor(nn.Module):
    """Project and budget DINOv3 dense patch tokens."""

    def __init__(
        self,
        *,
        projector: nn.Module,
        hidden_dim: int,
        max_nodes: int,
        budgeting: str,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.residual_projector = nn.Sequential(
            nn.Linear(12, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.max_nodes = int(max_nodes)
        self.budgeting = budgeting

    def forward(
        self,
        dinov3: DINOv3Output,
        *,
        image: torch.Tensor | None = None,
        image_content_mask: torch.Tensor | None = None,
    ) -> EvidenceSet:
        validate_patch_grid(
            token_count=dinov3.patch_tokens.shape[1],
            patch_grid=dinov3.patch_grid,
            context="visual_forensic",
        )
        descriptors, forensic_scores = forensic_patch_descriptors(
            image=image,
            image_content_mask=image_content_mask,
            grid_height=dinov3.patch_grid.height,
            grid_width=dinov3.patch_grid.width,
            fallback=dinov3.patch_tokens,
        )
        budgeted = budget_features(
            features=dinov3.patch_tokens,
            valid_mask=dinov3.patch_mask,
            max_nodes=self.max_nodes,
            strategy=self.budgeting,
            selection_scores=forensic_scores,
        )
        selected_descriptors = torch.gather(
            descriptors,
            1,
            budgeted.source_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, descriptors.shape[-1]),
        )
        selected_descriptors = selected_descriptors.masked_fill(~budgeted.mask.unsqueeze(-1), 0.0)
        projected = self.projector(budgeted.features) + self.residual_projector(selected_descriptors)
        centers, boxes, norm_centers, norm_boxes = patch_centers_and_boxes(
            dinov3.patch_grid,
            device=dinov3.patch_tokens.device,
            dtype=dinov3.patch_tokens.dtype,
        )
        batch_size = dinov3.patch_tokens.shape[0]
        full_centers = centers.unsqueeze(0).expand(batch_size, -1, -1)
        full_boxes = boxes.unsqueeze(0).expand(batch_size, -1, -1)
        full_norm_centers = norm_centers.unsqueeze(0).expand(batch_size, -1, -1)
        full_norm_boxes = norm_boxes.unsqueeze(0).expand(batch_size, -1, -1)
        return EvidenceSet(
            family="visual_forensic",
            features=projected,
            mask=budgeted.mask,
            source_indices=budgeted.source_indices,
            selection_scores=budgeted.selection_scores,
            spatial_centers=gather_spatial(centers, budgeted.source_indices),
            spatial_boxes=gather_spatial(boxes, budgeted.source_indices),
            normalized_centers=gather_spatial(norm_centers, budgeted.source_indices),
            normalized_boxes=gather_spatial(norm_boxes, budgeted.source_indices),
            full_features=dinov3.patch_tokens,
            full_mask=dinov3.patch_mask,
            full_source_indices=torch.arange(
                dinov3.patch_tokens.shape[1],
                device=dinov3.patch_tokens.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(dinov3.patch_tokens.shape[0], -1),
            full_spatial_centers=full_centers,
            full_spatial_boxes=full_boxes,
            full_normalized_centers=full_norm_centers,
            full_normalized_boxes=full_norm_boxes,
            provenance={
                "source": "dinov3.patch_tokens + rgb_high_frequency_residuals",
                "geometry_frame": "transformed_384",
                "budgeting": self.budgeting,
                "forensic_descriptors": ["rgb_mean", "local_variance", "high_pass", "gradient"],
            },
        )


def forensic_patch_descriptors(
    *,
    image: torch.Tensor | None,
    image_content_mask: torch.Tensor | None,
    grid_height: int,
    grid_width: int,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract target-independent RGB residual statistics for every patch."""

    if image is None:
        descriptors = fallback.new_zeros((fallback.shape[0], fallback.shape[1], 12))
        scores = torch.linalg.vector_norm(fallback.float(), dim=-1).to(fallback.dtype)
        return descriptors, scores
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must be [B,3,H,W] for forensic descriptors")
    work = image.float()
    if image_content_mask is None:
        valid = torch.ones_like(work[:, :1])
    else:
        if image_content_mask.shape != image.shape[0:1] + image.shape[2:4]:
            raise ValueError("image_content_mask must align with image [B,H,W]")
        valid = image_content_mask.to(device=work.device, dtype=work.dtype).unsqueeze(1)

    kernel = torch.ones((3, 1, 3, 3), device=work.device, dtype=work.dtype)
    local_sum = F.conv2d(work * valid, kernel, padding=1, groups=3)
    local_count = F.conv2d(valid, torch.ones((1, 1, 3, 3), device=work.device), padding=1).clamp_min(1.0)
    local_mean = local_sum / local_count
    residual = (work - local_mean) * valid
    local_sq_sum = F.conv2d(work.square() * valid, kernel, padding=1, groups=3)
    local_variance = (local_sq_sum / local_count - local_mean.square()).clamp_min(0.0) * valid

    dx = F.pad(work[..., 1:] - work[..., :-1], (0, 1, 0, 0))
    dy = F.pad(work[..., 1:, :] - work[..., :-1, :], (0, 0, 0, 1))
    valid_dx = valid * F.pad(valid[..., 1:] * valid[..., :-1], (0, 1, 0, 0))
    valid_dy = valid * F.pad(valid[..., 1:, :] * valid[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt((dx * valid_dx).square() + (dy * valid_dy).square() + 1e-12)

    signals = torch.cat([work * valid, local_variance, residual.abs(), gradient], dim=1)
    pooled = F.adaptive_avg_pool2d(signals, (grid_height, grid_width))
    descriptors = pooled.flatten(2).transpose(1, 2).to(dtype=fallback.dtype)
    if descriptors.shape[1] != fallback.shape[1]:
        raise ValueError("Forensic descriptor grid must align with DINO patch tokens")
    scores = (pooled[:, 6:9].mean(dim=1) + pooled[:, 9:12].mean(dim=1)).flatten(1)
    scores = scores.to(dtype=fallback.dtype)
    return descriptors, scores
