"""MTCNN candidate-face evidence from DINOv3 local patch features."""

from __future__ import annotations

import torch
from torch import nn

from src.models.heads.image_grounding import pool_candidate_patches
from src.models.outputs import DINOv3Output

from .budgeting import budget_features
from .outputs import EvidenceSet
from .spatial import gather_batched, normalize_boxes


class FaceEvidenceExtractor(nn.Module):
    """Pool DINOv3 features under legal MTCNN candidate boxes."""

    def __init__(
        self,
        *,
        projector: nn.Module,
        truncate: bool = False,
        max_nodes: int | None = None,
        budgeting: str = "ordered",
    ) -> None:
        super().__init__()
        self.projector = projector
        self.truncate = bool(truncate)
        self.max_nodes = None if max_nodes is None else int(max_nodes)
        self.budgeting = budgeting

    def forward(
        self,
        *,
        dinov3: DINOv3Output,
        candidate_boxes: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> EvidenceSet:
        if candidate_boxes.ndim != 3 or candidate_boxes.shape[-1] != 4:
            raise ValueError("candidate_boxes must be [B,F,4]")
        if candidate_mask.shape != candidate_boxes.shape[:2]:
            raise ValueError("candidate_mask must be [B,F]")
        candidate_mask = candidate_mask.to(torch.bool)
        pooled = pool_candidate_patches(
            patch_tokens=dinov3.patch_tokens,
            patch_grid=dinov3.patch_grid,
            candidate_boxes=candidate_boxes,
            candidate_mask=candidate_mask,
        )
        if self.truncate:
            budgeted = budget_features(
                features=pooled,
                valid_mask=candidate_mask,
                max_nodes=self.max_nodes,
                strategy=self.budgeting,
            )
            face_features = budgeted.features
            face_mask = budgeted.mask
            source_indices = budgeted.source_indices
        else:
            budgeted = budget_features(
                features=pooled,
                valid_mask=candidate_mask,
                max_nodes=None,
                strategy="ordered",
            )
            face_features = budgeted.features
            face_mask = budgeted.mask
            source_indices = budgeted.source_indices
        projected = self.projector(face_features)
        selected_boxes = gather_batched(candidate_boxes, source_indices)
        normalized_boxes = normalize_boxes(selected_boxes, patch_grid=dinov3.patch_grid)
        centers = torch.stack(
            [
                (selected_boxes[..., 0] + selected_boxes[..., 2]) * 0.5,
                (selected_boxes[..., 1] + selected_boxes[..., 3]) * 0.5,
            ],
            dim=-1,
        )
        normalized_centers = torch.stack(
            [
                centers[..., 0] / dinov3.patch_grid.image_width,
                centers[..., 1] / dinov3.patch_grid.image_height,
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        return EvidenceSet(
            family="face",
            features=projected,
            mask=face_mask,
            source_indices=source_indices,
            spatial_centers=centers,
            spatial_boxes=selected_boxes,
            normalized_centers=normalized_centers,
            normalized_boxes=normalized_boxes,
            full_features=pooled,
            full_mask=candidate_mask,
            full_source_indices=torch.arange(
                pooled.shape[1],
                device=pooled.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(pooled.shape[0], -1),
            provenance={
                "source": "mtcnn_boxes + dinov3.patch_tokens",
                "geometry_frame": "transformed_384",
                "truncate": self.truncate,
                "tiny_face_fallback": "nearest_patch_center",
            },
        )
