"""Phase-5 HERA attribution, derived-binary, and grounding losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from src.models.hera.outputs import HERADecisionOutput
from src.losses.box_grounding import dense_box_grounding_loss, normalized_target_boxes


@dataclass
class HERALossResult:
    """Structured HERA Phase-5 loss output."""

    total: torch.Tensor
    attribution: torch.Tensor
    binary: torch.Tensor
    image_grounding: torch.Tensor
    text_grounding: torch.Tensor
    evidence: torch.Tensor
    alignment: torch.Tensor
    router: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Detach losses for logging and smoke reports."""

        return {
            "total": float(self.total.detach().cpu()),
            "attribution": float(self.attribution.detach().cpu()),
            "binary": float(self.binary.detach().cpu()),
            "image_grounding": float(self.image_grounding.detach().cpu()),
            "text_grounding": float(self.text_grounding.detach().cpu()),
            "evidence": float(self.evidence.detach().cpu()),
            "alignment": float(self.alignment.detach().cpu()),
            "router": float(self.router.detach().cpu()),
        }


class HERAMultitaskLoss:
    """Compute Phase-5 losses with explicit hypothesis applicability."""

    def __init__(
        self,
        weights: Mapping[str, float],
        routing_config: Mapping[str, object] | None = None,
    ) -> None:
        self.lambda_attr = float(weights["lambda_attr"])
        self.lambda_bin = float(weights["lambda_bin"])
        self.lambda_img = float(weights["lambda_img"])
        self.lambda_txt = float(weights["lambda_txt"])
        self.lambda_ev = float(weights.get("lambda_ev", 0.0))
        self.lambda_align = float(weights.get("lambda_align", 0.0))
        image_config = weights.get("image_grounding", {})
        self.image_l1_weight = float(image_config.get("l1_weight", 1.0))
        self.image_giou_weight = float(image_config.get("giou_weight", 1.0))
        self.image_heatmap_weight = float(image_config.get("heatmap_weight", 0.25))
        regularization = (routing_config or {}).get("regularization", {})
        self.lambda_router = (
            float(regularization.get("weight", 0.0))
            if isinstance(regularization, Mapping) and bool(regularization.get("enabled", False))
            else 0.0
        )

    def __call__(self, output: HERADecisionOutput, batch: Mapping[str, torch.Tensor]) -> HERALossResult:
        """Compute L_attr + L_bin + L_img + L_txt + L_ev."""

        output.validate()
        atomic_target = batch["atomic_target"].to(device=output.atomic_logits.device, dtype=output.atomic_logits.dtype)
        binary_target = batch["binary_target"].to(device=output.binary_logit.device, dtype=output.binary_logit.dtype)
        if atomic_target.shape != output.atomic_logits.shape:
            raise ValueError("atomic_target must align with atomic_logits [B,K]")
        if binary_target.shape != output.binary_logit.shape:
            raise ValueError("binary_target must align with derived binary_logit [B]")

        attribution = F.binary_cross_entropy_with_logits(output.atomic_logits, atomic_target)
        if output.metadata.get("binary_aggregation") == "independent_head":
            binary = F.binary_cross_entropy_with_logits(output.binary_logit, binary_target)
        else:
            binary = target_consistent_noisy_or_loss(output.atomic_logits, atomic_target, binary_target)
        image_grounding = hypothesis_image_grounding_loss(
            output,
            batch,
            atomic_target=atomic_target,
            l1_weight=self.image_l1_weight,
            giou_weight=self.image_giou_weight,
            heatmap_weight=self.image_heatmap_weight,
        )
        text_grounding = hypothesis_text_grounding_loss(output, batch, atomic_target=atomic_target)
        evidence = evidence_sufficiency_loss(output)
        alignment = _optional_scalar_loss(output, "cross_modal_alignment_loss")
        router = _optional_scalar_loss(output, "router_regularization_loss")
        total = (
            self.lambda_attr * attribution
            + self.lambda_bin * binary
            + self.lambda_img * image_grounding
            + self.lambda_txt * text_grounding
            + self.lambda_ev * evidence
            + self.lambda_align * alignment
            + self.lambda_router * router
        )
        return HERALossResult(
            total=total,
            attribution=attribution,
            binary=binary,
            image_grounding=image_grounding,
            text_grounding=text_grounding,
            evidence=evidence,
            alignment=alignment,
            router=router,
        )


def target_consistent_noisy_or_loss(
    atomic_logits: torch.Tensor,
    atomic_target: torch.Tensor,
    binary_target: torch.Tensor,
) -> torch.Tensor:
    """Binary calibration without pushing absent hypotheses up on fake rows."""

    if atomic_logits.shape != atomic_target.shape:
        raise ValueError("atomic logits/targets must align")
    positive_or_real = (atomic_target > 0.5) | (binary_target.unsqueeze(1) < 0.5)
    consistent_logits = torch.where(positive_or_real, atomic_logits, atomic_logits.detach())
    log_not_fake = F.logsigmoid(-consistent_logits).sum(dim=1)
    binary_logit = torch.log((-torch.expm1(log_not_fake)).clamp_min(1e-12)) - log_not_fake
    return F.binary_cross_entropy_with_logits(binary_logit, binary_target)


def _optional_scalar_loss(output: HERADecisionOutput, name: str) -> torch.Tensor:
    value = getattr(output, name, None)
    if isinstance(value, torch.Tensor):
        return value
    return output.atomic_logits.sum() * 0.0


def image_grounding_applicability(
    atomic_target: torch.Tensor, indices: tuple[int, ...] = (0, 1)
) -> torch.Tensor:
    """Return applicability for the configured image-grounding hypotheses."""

    if atomic_target.ndim != 2 or not indices or max(indices) >= atomic_target.shape[1]:
        raise ValueError("atomic_target or image hypothesis indices are invalid")
    return atomic_target[:, list(indices)] > 0.5


def text_grounding_applicability(
    atomic_target: torch.Tensor, indices: tuple[int, ...] = (2, 3)
) -> torch.Tensor:
    """Return applicability for the configured text-grounding hypotheses."""

    if atomic_target.ndim != 2 or not indices or max(indices) >= atomic_target.shape[1]:
        raise ValueError("atomic_target or text hypothesis indices are invalid")
    return atomic_target[:, list(indices)] > 0.5


def hypothesis_image_grounding_loss(
    output: HERADecisionOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    atomic_target: torch.Tensor | None = None,
    l1_weight: float = 1.0,
    giou_weight: float = 1.0,
    heatmap_weight: float = 0.25,
) -> torch.Tensor:
    """Dense bbox/heatmap loss for the ground-truth FS/FA head during training."""

    logits = output.image_grounding_logits
    patch_mask = output.image_grounding_mask.to(device=logits.device, dtype=torch.bool)
    target_boxes = normalized_target_boxes(batch, device=logits.device, dtype=logits.dtype)
    if atomic_target is None:
        atomic_target = batch["atomic_target"].to(device=logits.device, dtype=logits.dtype)
    image_indices = tuple(
        int(value) for value in output.metadata.get("image_hypothesis_indices", (0, 1))
    )
    applies = image_grounding_applicability(
        atomic_target.to(device=logits.device), image_indices
    )
    annotation_available = batch.get("image_grounding_available")
    if annotation_available is not None:
        applies = applies & annotation_available.to(logits.device).to(torch.bool).unsqueeze(1)
    if patch_mask.shape != logits.shape[0:1] + logits.shape[2:3]:
        raise ValueError("image grounding mask shape mismatch")

    losses: list[torch.Tensor] = []
    for head_index in range(logits.shape[1]):
        valid_rows = applies[:, head_index]
        if not bool(valid_rows.any()):
            continue
        losses.append(
            dense_box_grounding_loss(
                predicted_boxes=output.image_grounding_boxes[:, head_index, :],
                patch_logits=logits[:, head_index, :],
                patch_mask=patch_mask,
                patch_boxes=output.image_grounding_patch_boxes,
                target_boxes=target_boxes,
                valid_rows=valid_rows,
                l1_weight=l1_weight,
                giou_weight=giou_weight,
                heatmap_weight=heatmap_weight,
            )
        )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def hypothesis_text_grounding_loss(
    output: HERADecisionOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    atomic_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE token loss for the relevant TS/TA grounding head only."""

    logits = output.text_grounding_logits
    text_mask = output.text_grounding_mask.to(device=logits.device, dtype=torch.bool)
    target = batch["text_grounding_target"].to(device=logits.device)
    if atomic_target is None:
        atomic_target = batch["atomic_target"].to(device=logits.device, dtype=logits.dtype)
    if target.shape != (logits.shape[0], logits.shape[-1]):
        raise ValueError("text_grounding_target must align with full text grounding logits [B,N_full_text]")
    text_indices = tuple(
        int(value) for value in output.metadata.get("text_hypothesis_indices", (2, 3))
    )
    applies = text_grounding_applicability(
        atomic_target.to(device=logits.device), text_indices
    )
    annotation_available = batch.get("text_grounding_available")
    if annotation_available is not None:
        applies = applies & annotation_available.to(logits.device).to(torch.bool).unsqueeze(1)

    losses: list[torch.Tensor] = []
    for head_index in range(logits.shape[1]):
        valid = applies[:, head_index].unsqueeze(1) & text_mask & (target >= 0)
        if not bool(valid.any()):
            continue
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits[:, head_index, :][valid],
                target[valid].to(dtype=logits.dtype),
            )
        )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def evidence_sufficiency_loss(output: HERADecisionOutput) -> torch.Tensor:
    """Simple optional consistency: high hypotheses should have local evidence support."""

    atomic = output.atomic_probabilities
    if output.evidence_support is not None:
        support = output.evidence_support
    else:
        image_support = _masked_max_probabilities(
            output.image_grounding_probabilities,
            output.image_grounding_mask,
        )
        text_support = _masked_max_probabilities(
            output.text_grounding_probabilities,
            output.text_grounding_mask,
        )
        support = torch.cat([image_support, text_support], dim=1)
    if support.shape != atomic.shape:
        raise ValueError("Evidence support must align with atomic probabilities [B,K]")
    return torch.relu(atomic - support).mean()


def _masked_max_probabilities(probabilities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 3 or probabilities.shape[1] < 1:
        raise ValueError("Hypothesis-conditioned grounding probabilities must be [B,K_grounding,N]")
    if mask.shape != (probabilities.shape[0], probabilities.shape[-1]):
        raise ValueError("Grounding mask must be [B,N]")
    masked = probabilities.masked_fill(~mask.to(torch.bool).unsqueeze(1), 0.0)
    if probabilities.shape[-1] == 0:
        return probabilities.new_zeros((probabilities.shape[0], 2))
    return masked.max(dim=-1).values
