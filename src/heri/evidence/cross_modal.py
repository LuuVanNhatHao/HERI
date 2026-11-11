"""Target-independent cross-modal semantic metadata."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .outputs import CrossModalMetadata, EvidenceSet


def cosine_similarity_metadata(
    *,
    visual: EvidenceSet,
    text: EvidenceSet,
    native_image_global: torch.Tensor | None = None,
    native_text_global: torch.Tensor | None = None,
) -> CrossModalMetadata:
    """Initial semantic relation in the common projected evidence space."""

    if visual.features.shape[-1] != text.features.shape[-1]:
        raise ValueError("Cross-modal cosine requires the shared evidence dimension")
    with torch.autocast(device_type=visual.features.device.type, enabled=False):
        visual_norm = F.normalize(visual.features.float(), p=2, dim=-1, eps=1e-8)
        text_norm = F.normalize(text.features.float(), p=2, dim=-1, eps=1e-8)
        similarity = torch.matmul(visual_norm, text_norm.transpose(1, 2))
    valid = visual.mask.unsqueeze(2) & text.mask.unsqueeze(1)
    similarity = similarity.masked_fill(~valid, 0.0)
    alignment_loss = _native_global_calibration_loss(
        similarity=similarity,
        valid=valid,
        native_image_global=native_image_global,
        native_text_global=native_text_global,
    )
    return CrossModalMetadata(
        similarity=similarity,
        visual_mask=visual.mask,
        text_mask=text.mask,
        visual_source_indices=visual.source_indices,
        text_source_indices=text.source_indices,
        method="cosine",
        feature_space="projected_evidence",
        alignment_loss=alignment_loss,
    )


def _native_global_calibration_loss(
    *,
    similarity: torch.Tensor,
    valid: torch.Tensor,
    native_image_global: torch.Tensor | None,
    native_text_global: torch.Tensor | None,
) -> torch.Tensor:
    """Calibrate mean text-to-image local similarity against a detached teacher."""

    # Similarity reductions and the native cosine target stay in FP32 even
    # when the surrounding encoder/evidence computation uses mixed precision.
    similarity = similarity.float()
    zero = similarity.sum() * 0.0
    if native_image_global is None or native_text_global is None:
        return zero
    if native_image_global.shape != native_text_global.shape or native_image_global.ndim != 2:
        return zero
    if similarity.shape[1] == 0 or similarity.shape[2] == 0:
        return zero
    target = F.cosine_similarity(
        native_image_global.float(), native_text_global.float(), dim=-1, eps=1e-8
    ).detach()
    masked = similarity.masked_fill(~valid, torch.finfo(similarity.dtype).min)
    best_visual_per_text = masked.max(dim=1).values
    valid_text = valid.any(dim=1)
    # Eq.29 averages one visual maximum per valid text position. Padding and
    # text positions without any valid visual counterpart do not enter the mean.
    local = best_visual_per_text.masked_fill(~valid_text, 0.0).sum(dim=1) / (
        valid_text.sum(dim=1).clamp_min(1)
    )
    usable = valid_text.any(dim=1)
    if not bool(usable.any()):
        return zero
    return F.smooth_l1_loss(local[usable], target[usable])
