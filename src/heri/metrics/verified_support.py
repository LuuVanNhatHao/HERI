"""Annotation-verified top-1 support, HERI Appendix D Eq.33."""

from __future__ import annotations

import torch


def verified_support_units(
    *,
    atomic_probabilities: torch.Tensor,
    atomic_targets: torch.Tensor,
    grounding_logits: torch.Tensor,
    grounding_mask: torch.Tensor,
    localization_targets: torch.Tensor,
    hypothesis_indices: tuple[int, ...],
    annotation_available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,K] verified-hit and eligible-unit masks for one modality.

    The population is correctly predicted positive hypotheses with applicable
    modality and valid annotation. Empty valid position sets count as misses,
    rather than disappearing from that population. No labels enter inference.
    """
    scores = atomic_probabilities.detach().float()
    targets = atomic_targets.to(device=scores.device)
    logits = grounding_logits.detach().float()
    mask = grounding_mask.to(device=scores.device, dtype=torch.bool)
    local_target = localization_targets.to(device=scores.device, dtype=torch.bool)
    if targets.shape != scores.shape or logits.ndim != 3:
        raise ValueError(
            "VSR requires aligned atomic scores/targets and [B,H,N] logits"
        )
    if logits.shape[:2] != (scores.shape[0], len(hypothesis_indices)):
        raise ValueError("VSR grounding heads must align with hypothesis indices")
    if (
        mask.shape != (scores.shape[0], logits.shape[-1])
        or local_target.shape != mask.shape
    ):
        raise ValueError(
            "VSR mask and annotation targets must align with dense positions"
        )
    if annotation_available.shape != (scores.shape[0],):
        raise ValueError("VSR annotation availability must be [B]")
    eligible = torch.zeros_like(scores, dtype=torch.bool)
    hits = torch.zeros_like(scores, dtype=torch.bool)
    available = annotation_available.to(device=scores.device, dtype=torch.bool)
    for head, hypothesis in enumerate(hypothesis_indices):
        valid = (
            (scores[:, hypothesis] >= 0.5) & (targets[:, hypothesis] > 0.5) & available
        )
        eligible[:, hypothesis] = valid
        if logits.shape[-1] == 0:
            continue
        values, indices = logits[:, head].masked_fill(~mask, -torch.inf).max(dim=-1)
        annotated = local_target.gather(1, indices[:, None]).squeeze(1)
        hits[:, hypothesis] = valid & mask.any(dim=-1) & (values >= 0.0) & annotated
    return hits, eligible
