"""SigLIP2 local visual-semantic evidence extraction."""

from __future__ import annotations

import torch
from torch import nn

from src.models.outputs import SigLIP2Output

from .budgeting import budget_features
from .outputs import EvidenceSet
from .spatial import gather_spatial, patch_centers_and_boxes, validate_patch_grid


class VisualSemanticEvidenceExtractor(nn.Module):
    """Project and budget SigLIP2 local image tokens."""

    def __init__(self, *, projector: nn.Module, max_nodes: int, budgeting: str) -> None:
        super().__init__()
        self.projector = projector
        self.max_nodes = int(max_nodes)
        self.budgeting = budgeting

    def forward(self, siglip2: SigLIP2Output) -> EvidenceSet:
        if siglip2.image_grid is None:
            raise ValueError("SigLIP2 image_grid is required for visual semantic evidence")
        image_tokens = siglip2.image_tokens
        mask = siglip2.image_mask
        if mask is None:
            mask = torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=image_tokens.device)
        validate_patch_grid(
            token_count=image_tokens.shape[1],
            patch_grid=siglip2.image_grid,
            context="visual_semantic",
        )
        budgeted = budget_features(
            features=image_tokens,
            valid_mask=mask,
            max_nodes=self.max_nodes,
            strategy=self.budgeting,
        )
        projected = self.projector(budgeted.features)
        centers, boxes, norm_centers, norm_boxes = patch_centers_and_boxes(
            siglip2.image_grid,
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )
        return EvidenceSet(
            family="visual_semantic",
            features=projected,
            mask=budgeted.mask,
            source_indices=budgeted.source_indices,
            spatial_centers=gather_spatial(centers, budgeted.source_indices),
            spatial_boxes=gather_spatial(boxes, budgeted.source_indices),
            normalized_centers=gather_spatial(norm_centers, budgeted.source_indices),
            normalized_boxes=gather_spatial(norm_boxes, budgeted.source_indices),
            full_features=image_tokens,
            full_mask=mask,
            full_source_indices=torch.arange(
                image_tokens.shape[1],
                device=image_tokens.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(image_tokens.shape[0], -1),
            provenance={
                "source": "siglip2.image_tokens",
                "geometry_frame": "transformed_384",
                "budgeting": self.budgeting,
            },
        )
