"""Protocol-clean dense bounding-box supervision shared by HERA and baselines."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def normalized_target_boxes(
    batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize transformed-space DGM4 xyxy targets using the actual model image size."""

    targets = batch["fake_image_box_target"].to(device=device, dtype=dtype)
    image = batch["image"]
    if image.ndim != 4:
        raise ValueError("batch.image must be [B,C,H,W]")
    height, width = int(image.shape[-2]), int(image.shape[-1])
    scale = targets.new_tensor([width, height, width, height])
    return (targets / scale).clamp(0.0, 1.0)


def dense_box_grounding_loss(
    *,
    predicted_boxes: torch.Tensor,
    patch_logits: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    valid_rows: torch.Tensor,
    l1_weight: float = 1.0,
    giou_weight: float = 1.0,
    heatmap_weight: float = 0.25,
) -> torch.Tensor:
    """L1 + generalized-IoU + dense patch supervision for aligned rows."""

    if predicted_boxes.ndim != 2 or predicted_boxes.shape[-1] != 4:
        raise ValueError("predicted_boxes must be [B,4]")
    if target_boxes.shape != predicted_boxes.shape:
        raise ValueError("target_boxes must align with predicted_boxes")
    if patch_logits.shape != patch_mask.shape:
        raise ValueError("patch_logits and patch_mask must align")
    if patch_boxes.shape != patch_logits.shape + (4,):
        raise ValueError("patch_boxes must be [B,N_patch,4]")
    valid_rows = valid_rows.to(device=predicted_boxes.device, dtype=torch.bool)
    if not bool(valid_rows.any()):
        return predicted_boxes.sum() * 0.0 + patch_logits.sum() * 0.0

    pred = predicted_boxes[valid_rows]
    target = target_boxes[valid_rows]
    l1 = F.smooth_l1_loss(pred, target)
    giou = aligned_generalized_box_iou(pred, target)
    box_loss = float(l1_weight) * l1 + float(giou_weight) * (1.0 - giou).mean()

    selected_logits = patch_logits[valid_rows]
    selected_mask = patch_mask[valid_rows].to(torch.bool)
    selected_patch_boxes = patch_boxes[valid_rows]
    heatmap_targets = patch_overlap_targets(selected_patch_boxes, target)
    if bool(selected_mask.any()) and float(heatmap_weight) > 0:
        heatmap = F.binary_cross_entropy_with_logits(
            selected_logits[selected_mask],
            heatmap_targets[selected_mask].to(selected_logits.dtype),
        )
        box_loss = box_loss + float(heatmap_weight) * heatmap
    return box_loss


def patch_overlap_targets(patch_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Mark every patch intersecting the target, including tiny manipulations."""

    intersection_lt = torch.maximum(patch_boxes[..., :2], target_boxes[:, None, :2])
    intersection_rb = torch.minimum(patch_boxes[..., 2:], target_boxes[:, None, 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp_min(0.0)
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
    return (intersection_area > 0).to(patch_boxes.dtype)


def aligned_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU for aligned normalized xyxy boxes."""

    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[-1] != 4:
        raise ValueError("aligned boxes must both be [N,4]")
    intersection_lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    intersection_rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp_min(0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    union = (area1 + area2 - intersection).clamp_min(1e-8)
    iou = intersection / union
    enclosing_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclosing_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp_min(0.0)
    enclosing = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp_min(1e-8)
    return iou - (enclosing - union) / enclosing


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
    return wh[:, 0] * wh[:, 1]
