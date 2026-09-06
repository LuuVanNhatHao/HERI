"""Conventional multitask losses for Phase-2 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from src.models.outputs import BaselineOutput
from src.losses.box_grounding import dense_box_grounding_loss, normalized_target_boxes


@dataclass
class BaselineLossResult:
    """Structured baseline loss output."""

    total: torch.Tensor
    binary: torch.Tensor
    attribution: torch.Tensor
    image_grounding: torch.Tensor
    text_grounding: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Detach losses for logging."""

        return {
            "total": float(self.total.detach().cpu()),
            "binary": float(self.binary.detach().cpu()),
            "attribution": float(self.attribution.detach().cpu()),
            "image_grounding": float(self.image_grounding.detach().cpu()),
            "text_grounding": float(self.text_grounding.detach().cpu()),
        }


class BaselineMultitaskLoss:
    """Compute conventional B0/B1 multitask losses with applicability masks."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        self.lambda_binary = float(weights["binary"])
        self.lambda_attribution = float(weights["attribution"])
        self.lambda_image = float(weights["image_grounding"])
        self.lambda_text = float(weights["text_grounding"])

    def __call__(self, output: BaselineOutput, batch: Mapping[str, torch.Tensor]) -> BaselineLossResult:
        if output.metadata.get("binary_aggregation") == "noisy_or":
            from src.losses.hera_components import target_consistent_noisy_or_loss

            binary = target_consistent_noisy_or_loss(
                output.atomic_logits,
                batch["atomic_target"].to(output.atomic_logits),
                batch["binary_target"].to(output.binary_logit),
            )
        else:
            binary = F.binary_cross_entropy_with_logits(
                output.binary_logit,
                batch["binary_target"].to(output.binary_logit.dtype),
            )
        attribution = F.binary_cross_entropy_with_logits(
            output.atomic_logits,
            batch["atomic_target"].to(output.atomic_logits.dtype),
        )
        image_grounding = masked_image_bbox_loss(output, batch)
        text_grounding = masked_text_grounding_loss(output, batch)
        total = (
            self.lambda_binary * binary
            + self.lambda_attribution * attribution
            + self.lambda_image * image_grounding
            + self.lambda_text * text_grounding
        )
        return BaselineLossResult(
            total=total,
            binary=binary,
            attribution=attribution,
            image_grounding=image_grounding,
            text_grounding=text_grounding,
        )


def masked_image_bbox_loss(
    output: BaselineOutput,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Dense bbox/heatmap loss using only raw-image patch predictions."""

    logits = output.image_grounding_logits
    applies = batch["has_image_manipulation"].to(logits.device).to(torch.bool)
    if "image_grounding_available" in batch:
        applies = applies & batch["image_grounding_available"].to(logits.device).to(torch.bool)
    targets = normalized_target_boxes(batch, device=logits.device, dtype=logits.dtype)
    return dense_box_grounding_loss(
        predicted_boxes=output.image_grounding_boxes,
        patch_logits=logits,
        patch_mask=output.image_grounding_mask.to(logits.device),
        patch_boxes=output.image_grounding_patch_boxes,
        target_boxes=targets,
        valid_rows=applies,
    )


def masked_text_grounding_loss(
    output: BaselineOutput,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """BCE token loss over valid DeBERTa subwords for text-manipulated samples."""

    logits = output.text_grounding_logits
    target = batch["text_grounding_target"].to(logits.device)
    mask = output.text_grounding_mask.to(logits.device).to(torch.bool)
    applies = batch["has_text_manipulation"].to(logits.device).to(torch.bool).unsqueeze(1)
    if "text_grounding_available" in batch:
        applies = applies & batch["text_grounding_available"].to(logits.device).to(torch.bool).unsqueeze(1)
    valid = mask & applies & (target >= 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[valid],
        target[valid].to(logits.dtype),
    )
