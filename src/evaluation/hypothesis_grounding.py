"""Adapters from hypothesis-conditioned grounding to metric-ready tensors."""

from __future__ import annotations

import torch

from src.models.hera.outputs import HERADecisionOutput


def select_applicable_image_grounding(
    output: HERADecisionOutput,
    atomic_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,N_face] logits and [B] applicability for FS/FA evaluation."""

    if atomic_target.ndim != 2 or atomic_target.shape[1] != 4:
        raise ValueError("atomic_target must be [B,4]")
    logits = output.image_grounding_logits
    if atomic_target.shape[0] != logits.shape[0]:
        raise ValueError("atomic_target batch size mismatch")
    selected = logits.new_zeros((logits.shape[0], logits.shape[-1]))
    fs = atomic_target[:, 0].to(device=logits.device) > 0.5
    fa = atomic_target[:, 1].to(device=logits.device) > 0.5
    selected[fs] = logits[fs, 0, :]
    selected[fa] = logits[fa, 1, :]
    return selected, fs | fa


def select_applicable_text_grounding(
    output: HERADecisionOutput,
    atomic_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,N_full_text] logits and [B] applicability for TS/TA evaluation."""

    if atomic_target.ndim != 2 or atomic_target.shape[1] != 4:
        raise ValueError("atomic_target must be [B,4]")
    logits = output.text_grounding_logits
    if atomic_target.shape[0] != logits.shape[0]:
        raise ValueError("atomic_target batch size mismatch")
    selected = logits.new_zeros((logits.shape[0], logits.shape[-1]))
    ts = atomic_target[:, 2].to(device=logits.device) > 0.5
    ta = atomic_target[:, 3].to(device=logits.device) > 0.5
    selected[ts] = logits[ts, 0, :]
    selected[ta] = logits[ta, 1, :]
    return selected, ts | ta
