"""Masked loss composition for binary-only HERA benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .contracts import SignalContract
from .model import BinaryBenchmarkOutput


@dataclass
class BinaryBenchmarkLossResult:
    total: torch.Tensor
    binary: torch.Tensor
    alignment: torch.Tensor
    router: torch.Tensor
    subtype: torch.Tensor
    text_label: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in (
                "total",
                "binary",
                "alignment",
                "router",
                "subtype",
                "text_label",
            )
        }


class BinaryBenchmarkLoss:
    """Apply only supervision that exists and is enabled in the dataset contract."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.signals = SignalContract.from_config(config)
        self.weights = {name: float(value) for name, value in config["loss"].items()}

    def __call__(
        self,
        output: BinaryBenchmarkOutput,
        batch: Mapping[str, torch.Tensor],
    ) -> BinaryBenchmarkLossResult:
        output.validate()
        anchor = output.binary_logit
        binary_target = batch["binary_target"].to(
            device=anchor.device, dtype=anchor.dtype
        )
        binary_mask = batch["binary_available"].to(
            device=anchor.device, dtype=torch.bool
        )
        binary = _masked_bce(
            output.binary_logit, binary_target, binary_mask, required=True
        )
        alignment = _optional_scalar(output.cross_modal_alignment_loss, anchor)
        router = _optional_scalar(output.router_regularization_loss, anchor)

        subtype = anchor.sum() * 0.0
        if self.signals.subtype:
            if output.subtype_logits is None:
                raise RuntimeError(
                    "Subtype supervision was enabled without a subtype head"
                )
            target = batch["subtype_target"].to(device=anchor.device, dtype=torch.long)
            mask = batch["subtype_available"].to(device=anchor.device, dtype=torch.bool)
            if not bool(mask.any()):
                raise ValueError(
                    "Subtype supervision is enabled but this batch has no subtype labels"
                )
            subtype = F.cross_entropy(output.subtype_logits[mask], target[mask])

        text_label = anchor.sum() * 0.0
        if self.signals.text_label:
            if output.text_label_logit is None:
                raise RuntimeError(
                    "text_label supervision was enabled without a text_label head"
                )
            target = batch["text_label_target"].to(
                device=anchor.device, dtype=anchor.dtype
            )
            mask = batch["text_label_available"].to(
                device=anchor.device, dtype=torch.bool
            )
            text_label = _masked_bce(
                output.text_label_logit, target, mask, required=True
            )

        total = (
            self.weights["lambda_bin"] * binary
            + self.weights["lambda_align"] * alignment
            + self.weights["lambda_router"] * router
            + self.weights["lambda_subtype"] * subtype
            + self.weights["lambda_text_label"] * text_label
        )
        return BinaryBenchmarkLossResult(
            total=total,
            binary=binary,
            alignment=alignment,
            router=router,
            subtype=subtype,
            text_label=text_label,
        )


def _masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    required: bool,
) -> torch.Tensor:
    if logits.shape != targets.shape or mask.shape != logits.shape:
        raise ValueError(
            "Masked BCE logits, targets, and mask must have identical shapes"
        )
    if not bool(mask.any()):
        if required:
            raise ValueError(
                "Required binary signal is unavailable for the entire batch"
            )
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def _optional_scalar(value: torch.Tensor | None, anchor: torch.Tensor) -> torch.Tensor:
    if value is None:
        return anchor.sum() * 0.0
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError("Optional structural loss must be a finite scalar")
    return value
