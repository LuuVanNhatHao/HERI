"""HERA attribution, derived-binary, and grounding losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from heri.models.heri.outputs import HERADecisionOutput
from heri.losses.box_grounding import dense_box_grounding_loss, normalized_target_boxes


@dataclass
class HERALossResult:
    """Structured HERA loss output."""

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
    """Compute losses with explicit hypothesis applicability."""

    def __init__(
        self,
        weights: Mapping[str, object],
        routing_config: Mapping[str, object] | None = None,
    ) -> None:
        self.lambda_attr = float(weights["lambda_attr"])
        self.lambda_bin = float(weights["lambda_bin"])
        self.lambda_img = float(weights["lambda_img"])
        self.lambda_txt = float(weights["lambda_txt"])
        self.lambda_ev = float(weights.get("lambda_ev", 0.0))
        self.lambda_align = float(weights.get("lambda_align", 0.0))
        evidence_config = weights.get("evidence_sufficiency", {})
        if not isinstance(evidence_config, Mapping):
            raise ValueError("loss.evidence_sufficiency must be a mapping")
        self.evidence_sufficiency_protocol = str(
            evidence_config.get("protocol", "per_modality_v1")
        )
        self.evidence_sufficiency_aggregation = str(
            evidence_config.get("aggregation", "per_modality")
        )
        self.evidence_sufficiency_validity = str(
            evidence_config.get(
                "validity", "positive_hypothesis_and_valid_annotation"
            )
        )
        self.evidence_sufficiency_empty_policy = str(
            evidence_config.get("empty_policy", "differentiable_zero")
        )
        if self.evidence_sufficiency_aggregation not in {
            "per_modality",
            "any_modality_or",
        }:
            raise ValueError(
                "Evidence sufficiency aggregation must be per_modality or "
                "any_modality_or"
            )
        expected_protocol = {
            "per_modality": "per_modality_v1",
            "any_modality_or": "any_modality_or_v1",
        }[self.evidence_sufficiency_aggregation]
        expected_validity = {
            "per_modality": "positive_hypothesis_and_valid_annotation",
            "any_modality_or": "unmasked_legacy",
        }[self.evidence_sufficiency_aggregation]
        if self.evidence_sufficiency_protocol != expected_protocol:
            raise ValueError(
                "Evidence sufficiency protocol/aggregation mismatch: expected "
                f"{expected_protocol!r}"
            )
        if self.evidence_sufficiency_validity != expected_validity:
            raise ValueError(
                "Evidence sufficiency validity/aggregation mismatch: expected "
                f"{expected_validity!r}"
            )
        if self.evidence_sufficiency_empty_policy != "differentiable_zero":
            raise ValueError(
                "Evidence sufficiency empty_policy must be differentiable_zero"
            )
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
        evidence = evidence_sufficiency_loss(
            output,
            batch,
            atomic_target=atomic_target,
            aggregation=self.evidence_sufficiency_aggregation,
        )
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
    consistent_logits = torch.where(positive_or_real, atomic_logits, atomic_logits.detach()).float()
    from heri.models.heads.noisy_or import StableNoisyOR
    binary_logit = StableNoisyOR()(consistent_logits).binary_logit
    return F.binary_cross_entropy_with_logits(binary_logit, binary_target.float())


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

    # Eq.26 averages valid (sample,hypothesis) entries, not equally weighted
    # per-head means, which differ when head populations are imbalanced.
    batch_size, heads, patches = logits.shape
    return dense_box_grounding_loss(
        predicted_boxes=output.image_grounding_boxes.reshape(-1, 4).float(),
        patch_logits=logits.reshape(-1, patches).float(),
        patch_mask=patch_mask[:, None, :].expand(-1, heads, -1).reshape(-1, patches),
        patch_boxes=output.image_grounding_patch_boxes[:, None, :, :].expand(-1, heads, -1, -1).reshape(-1, patches, 4).float(),
        target_boxes=target_boxes[:, None, :].expand(-1, heads, -1).reshape(-1, 4).float(),
        valid_rows=applies.reshape(-1),
        l1_weight=l1_weight,
        giou_weight=giou_weight,
        heatmap_weight=heatmap_weight,
    )


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

    valid = applies.unsqueeze(-1) & text_mask.unsqueeze(1) & (target >= 0).unsqueeze(1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits.float()[valid],
        target[:, None, :].expand_as(logits)[valid].float(),
    )


def evidence_sufficiency_loss(
    output: HERADecisionOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    atomic_target: torch.Tensor | None = None,
    aggregation: str = "per_modality",
) -> torch.Tensor:
    """Require every applicable annotated modality to support its hypothesis."""

    atomic = output.atomic_probabilities
    if atomic_target is None:
        atomic_target = batch["atomic_target"].to(
            device=atomic.device,
            dtype=atomic.dtype,
        )
    else:
        atomic_target = atomic_target.to(device=atomic.device, dtype=atomic.dtype)
    if atomic_target.shape != atomic.shape:
        raise ValueError("atomic_target must align with atomic probabilities [B,K]")
    support = output.evidence_support_by_modality
    modality_mask = output.evidence_support_modality_mask
    if support is None or modality_mask is None:
        raise ValueError(
            "Per-modality evidence support is required by the current loss protocol"
        )
    if support.shape != atomic.shape + (2,):
        raise ValueError("evidence_support_by_modality must be [B,K,2]")
    if modality_mask.shape != (atomic.shape[1], 2):
        raise ValueError("evidence_support_modality_mask must be [K,2]")

    if aggregation == "any_modality_or":
        applicable_support = support * modality_mask.unsqueeze(0).to(support.dtype)
        union = 1.0 - torch.prod(1.0 - applicable_support, dim=-1)
        return torch.relu(atomic - union).mean()
    if aggregation != "per_modality":
        raise ValueError(f"Unknown evidence sufficiency aggregation: {aggregation!r}")

    image_available = _annotation_availability(
        batch,
        "image_grounding_available",
        batch_size=atomic.shape[0],
        device=atomic.device,
    )
    text_available = _annotation_availability(
        batch,
        "text_grounding_available",
        batch_size=atomic.shape[0],
        device=atomic.device,
    )
    modality_available = torch.stack((image_available, text_available), dim=1)
    node_available = torch.stack(
        (
            output.image_grounding_mask.any(dim=1),
            output.text_grounding_mask.any(dim=1),
        ),
        dim=1,
    ).to(device=atomic.device)
    valid = (
        (atomic_target > 0.5).unsqueeze(-1)
        & modality_mask.unsqueeze(0)
        & modality_available.unsqueeze(1)
        & node_available.unsqueeze(1)
    )
    if not bool(valid.any()):
        return output.atomic_logits.sum() * 0.0
    penalties = torch.relu(atomic.unsqueeze(-1) - support)
    return penalties[valid].mean()


def _annotation_availability(
    batch: Mapping[str, torch.Tensor],
    key: str,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if value.shape != (batch_size,):
        raise ValueError(f"{key} must be [B]")
    return value.to(device=device, dtype=torch.bool)
