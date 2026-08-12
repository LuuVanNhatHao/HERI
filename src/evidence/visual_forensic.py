"""DINOv3 dense visual-forensic evidence extraction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.outputs import DINOv3Output

from .budgeting import budget_features
from .outputs import EvidenceSet
from .spatial import gather_spatial, patch_centers_and_boxes, validate_patch_grid


class VisualForensicEvidenceExtractor(nn.Module):
    """Project and budget DINOv3 dense patch tokens."""

    def __init__(self, *, projector: nn.Module, max_nodes: int, budgeting: str) -> None:
        super().__init__()
        self.projector = projector
        self.max_nodes = int(max_nodes)
        self.budgeting = budgeting

    def forward(self, dinov3: DINOv3Output) -> EvidenceSet:
        validate_patch_grid(
            token_count=dinov3.patch_tokens.shape[1],
            patch_grid=dinov3.patch_grid,
            context="visual_forensic",
        )
        budgeted = budget_features(
            features=dinov3.patch_tokens,
            valid_mask=dinov3.patch_mask,
            max_nodes=self.max_nodes,
            strategy=self.budgeting,
        )
        projected = self.projector(budgeted.features)
        centers, boxes, norm_centers, norm_boxes = patch_centers_and_boxes(
            dinov3.patch_grid,
            device=dinov3.patch_tokens.device,
            dtype=dinov3.patch_tokens.dtype,
        )
        return EvidenceSet(
            family="visual_forensic",
            features=projected,
            mask=budgeted.mask,
            source_indices=budgeted.source_indices,
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
            provenance={
                "source": "dinov3.patch_tokens",
                "geometry_frame": "transformed_384",
                "budgeting": self.budgeting,
            },
        )
