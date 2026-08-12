"""Stable attribution-derived Noisy-OR authenticity decision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.graph.schema import HYPOTHESIS_ORDER


@dataclass
class NoisyOROutput:
    """Structured zero-parameter Noisy-OR output."""

    atomic_probabilities: torch.Tensor
    binary_probability: torch.Tensor
    binary_logit: torch.Tensor
    real_probability: torch.Tensor
    log_real: torch.Tensor
    log_fake: torch.Tensor


class StableNoisyOR(nn.Module):
    """Derive Fake probability only from FS/FA/TS/TA logits."""

    def forward(self, atomic_logits: torch.Tensor) -> NoisyOROutput:
        """Compute stable Noisy-OR outputs from atomic logits [B,4]."""

        if atomic_logits.ndim != 2 or atomic_logits.shape[1] != len(HYPOTHESIS_ORDER):
            raise ValueError(f"atomic_logits must be [B,4], got {tuple(atomic_logits.shape)}")
        if not torch.isfinite(atomic_logits).all():
            raise FloatingPointError("atomic_logits contain non-finite values")

        compute_logits = _noisy_or_compute_logits(atomic_logits)
        atomic_probabilities = torch.sigmoid(compute_logits)
        log_real = F.logsigmoid(-compute_logits).sum(dim=1)
        tiny_negative = torch.full_like(log_real, -torch.finfo(log_real.dtype).tiny)
        safe_log_real = torch.where(log_real < 0, log_real, tiny_negative)
        log_fake = _log1mexp(safe_log_real)
        binary_logit = log_fake - safe_log_real
        real_probability = torch.exp(safe_log_real)
        binary_probability = -torch.expm1(safe_log_real)
        return NoisyOROutput(
            atomic_probabilities=atomic_probabilities,
            binary_probability=binary_probability,
            binary_logit=binary_logit,
            real_probability=real_probability,
            log_real=safe_log_real,
            log_fake=log_fake,
        )


def noisy_or_from_probabilities(atomic_probabilities: torch.Tensor) -> torch.Tensor:
    """Reference probability-space Noisy-OR for tests and diagnostics."""

    if atomic_probabilities.ndim != 2 or atomic_probabilities.shape[1] != len(HYPOTHESIS_ORDER):
        raise ValueError("atomic_probabilities must be [B,4]")
    if bool(((atomic_probabilities < 0) | (atomic_probabilities > 1)).any()):
        raise ValueError("atomic_probabilities must be in [0,1]")
    return 1.0 - torch.prod(1.0 - atomic_probabilities, dim=1)


def _noisy_or_compute_logits(atomic_logits: torch.Tensor) -> torch.Tensor:
    """Use FP32 math for low-precision autocast logits while preserving gradients."""

    if atomic_logits.dtype in {torch.float16, torch.bfloat16}:
        return atomic_logits.float()
    return atomic_logits


def _log1mexp(log_value: torch.Tensor) -> torch.Tensor:
    """Return log(1-exp(log_value)) for log_value <= 0."""

    threshold = -0.6931471805599453
    result = torch.empty_like(log_value)
    lower = log_value < threshold
    if bool(lower.any()):
        result[lower] = torch.log1p(-torch.exp(log_value[lower]))
    if bool((~lower).any()):
        result[~lower] = torch.log(-torch.expm1(log_value[~lower]))
    return result
