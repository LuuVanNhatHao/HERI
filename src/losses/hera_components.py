"""Phase-5 HERA attribution, derived-binary, and grounding losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from src.models.hera.outputs import HERADecisionOutput


@dataclass
class HERALossResult:
    """Structured HERA Phase-5 loss output."""

    total: torch.Tensor
    attribution: torch.Tensor
    binary: torch.Tensor
    image_grounding: torch.Tensor
    text_grounding: torch.Tensor
    evidence: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """Detach losses for logging and smoke reports."""

        return {
            "total": float(self.total.detach().cpu()),
            "attribution": float(self.attribution.detach().cpu()),
            "binary": float(self.binary.detach().cpu()),
            "image_grounding": float(self.image_grounding.detach().cpu()),
            "text_grounding": float(self.text_grounding.detach().cpu()),
            "evidence": float(self.evidence.detach().cpu()),
        }


class HERAMultitaskLoss:
    """Compute Phase-5 losses with explicit hypothesis applicability."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        self.lambda_attr = float(weights["lambda_attr"])
        self.lambda_bin = float(weights["lambda_bin"])
        self.lambda_img = float(weights["lambda_img"])
        self.lambda_txt = float(weights["lambda_txt"])
        self.lambda_ev = float(weights.get("lambda_ev", 0.0))

    def __call__(self, output: HERADecisionOutput, batch: Mapping[str, torch.Tensor]) -> HERALossResult:
        """Compute L_attr + L_bin + L_img + L_txt + L_ev."""

        output.validate()
        atomic_target = batch["atomic_target"].to(device=output.atomic_logits.device, dtype=output.atomic_logits.dtype)
        binary_target = batch["binary_target"].to(device=output.binary_logit.device, dtype=output.binary_logit.dtype)
        if atomic_target.shape != output.atomic_logits.shape:
            raise ValueError("atomic_target must align with atomic_logits [B,4]")
        if binary_target.shape != output.binary_logit.shape:
            raise ValueError("binary_target must align with derived binary_logit [B]")

        attribution = F.binary_cross_entropy_with_logits(output.atomic_logits, atomic_target)
        binary = F.binary_cross_entropy_with_logits(output.binary_logit, binary_target)
        image_grounding = hypothesis_image_grounding_loss(output, batch, atomic_target=atomic_target)
        text_grounding = hypothesis_text_grounding_loss(output, batch, atomic_target=atomic_target)
        evidence = evidence_sufficiency_loss(output)
        total = (
            self.lambda_attr * attribution
            + self.lambda_bin * binary
            + self.lambda_img * image_grounding
            + self.lambda_txt * text_grounding
            + self.lambda_ev * evidence
        )
        return HERALossResult(
            total=total,
            attribution=attribution,
            binary=binary,
            image_grounding=image_grounding,
            text_grounding=text_grounding,
            evidence=evidence,
        )


def image_grounding_applicability(atomic_target: torch.Tensor) -> torch.Tensor:
    """Return [B,2] applicability: column 0=FS, column 1=FA."""

    if atomic_target.ndim != 2 or atomic_target.shape[1] != 4:
        raise ValueError("atomic_target must be [B,4]")
    return atomic_target[:, :2] > 0.5


def text_grounding_applicability(atomic_target: torch.Tensor) -> torch.Tensor:
    """Return [B,2] applicability: column 0=TS, column 1=TA."""

    if atomic_target.ndim != 2 or atomic_target.shape[1] != 4:
        raise ValueError("atomic_target must be [B,4]")
    return atomic_target[:, 2:4] > 0.5


def hypothesis_image_grounding_loss(
    output: HERADecisionOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    atomic_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Candidate-selection CE for the relevant FS/FA grounding head only."""

    logits = output.image_grounding_logits
    face_mask = output.image_grounding_mask.to(device=logits.device, dtype=torch.bool)
    target_index = batch["manipulated_face_target_index"].to(device=logits.device, dtype=torch.long)
    if atomic_target is None:
        atomic_target = batch["atomic_target"].to(device=logits.device, dtype=logits.dtype)
    applies = image_grounding_applicability(atomic_target.to(device=logits.device))
    if target_index.shape != (logits.shape[0],):
        raise ValueError("manipulated_face_target_index must be [B]")
    if face_mask.shape != logits.shape[0:1] + logits.shape[2:3]:
        raise ValueError("image grounding mask shape mismatch")

    losses: list[torch.Tensor] = []
    for head_index in range(2):
        valid_rows = applies[:, head_index] & (target_index >= 0) & (target_index < logits.shape[-1])
        if not bool(valid_rows.any()):
            continue
        selected_mask = face_mask[valid_rows]
        selected_target = target_index[valid_rows]
        target_is_valid = selected_mask.gather(1, selected_target.unsqueeze(1)).squeeze(1)
        if not bool(target_is_valid.any()):
            continue
        selected_logits = logits[valid_rows, head_index, :][target_is_valid]
        selected_mask = selected_mask[target_is_valid]
        selected_target = selected_target[target_is_valid]
        losses.append(F.cross_entropy(selected_logits.masked_fill(~selected_mask, -1e4), selected_target))
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
    applies = text_grounding_applicability(atomic_target.to(device=logits.device))

    losses: list[torch.Tensor] = []
    for head_index in range(2):
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
        raise ValueError("Evidence support must align with atomic probabilities [B,4]")
    return torch.relu(atomic - support).mean()


def _masked_max_probabilities(probabilities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 3 or probabilities.shape[1] != 2:
        raise ValueError("Hypothesis-conditioned grounding probabilities must be [B,2,N]")
    if mask.shape != (probabilities.shape[0], probabilities.shape[-1]):
        raise ValueError("Grounding mask must be [B,N]")
    masked = probabilities.masked_fill(~mask.to(torch.bool).unsqueeze(1), 0.0)
    if probabilities.shape[-1] == 0:
        return probabilities.new_zeros((probabilities.shape[0], 2))
    return masked.max(dim=-1).values
