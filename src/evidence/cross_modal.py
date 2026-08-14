"""Target-independent cross-modal semantic metadata."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .outputs import CrossModalMetadata, EvidenceSet
from .spatial import gather_batched


def cosine_similarity_metadata(
    *,
    visual: EvidenceSet,
    text: EvidenceSet,
    native_image_global: torch.Tensor | None = None,
    native_text_global: torch.Tensor | None = None,
) -> CrossModalMetadata:
    """Compute local SigLIP similarity and calibrate it to native joint globals."""

    visual_source = _selected_source_features(visual)
    text_source = _selected_source_features(text)
    if visual_source.shape[-1] == text_source.shape[-1]:
        visual_features = visual_source
        text_features = text_source
        feature_space = "siglip_source_tokens"
    elif visual.features.shape[-1] == text.features.shape[-1]:
        # Compatibility fallback for heterogeneous test/dummy backbones. The
        # explicit native-global calibration below supervises this space when
        # joint SigLIP global embeddings are available.
        visual_features = visual.features
        text_features = text.features
        feature_space = "calibrated_evidence_projection"
    else:
        raise ValueError("Cross-modal cosine requires a matched source or evidence dimension")
    visual_norm = F.normalize(visual_features.float(), p=2, dim=-1, eps=1e-12)
    text_norm = F.normalize(text_features.float(), p=2, dim=-1, eps=1e-12)
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
        similarity=similarity.to(dtype=visual.features.dtype),
        visual_mask=visual.mask,
        text_mask=text.mask,
        visual_source_indices=visual.source_indices,
        text_source_indices=text.source_indices,
        method="cosine",
        feature_space=feature_space,
        alignment_loss=alignment_loss,
    )


def _selected_source_features(evidence: EvidenceSet) -> torch.Tensor:
    if evidence.full_features is None:
        return evidence.features
    return gather_batched(evidence.full_features, evidence.source_indices)


def _native_global_calibration_loss(
    *,
    similarity: torch.Tensor,
    valid: torch.Tensor,
    native_image_global: torch.Tensor | None,
    native_text_global: torch.Tensor | None,
) -> torch.Tensor:
    zero = similarity.sum() * 0.0
    if native_image_global is None or native_text_global is None:
        return zero
    if native_image_global.shape != native_text_global.shape or native_image_global.ndim != 2:
        return zero
    target = F.cosine_similarity(
        native_image_global.float(), native_text_global.float(), dim=-1, eps=1e-12
    ).detach()
    masked = similarity.masked_fill(~valid, torch.finfo(similarity.dtype).min)
    best_visual_per_text = masked.max(dim=1).values
    valid_text = valid.any(dim=1)
    local = (best_visual_per_text.masked_fill(~valid_text, 0.0).sum(dim=1) /
             valid_text.sum(dim=1).clamp_min(1))
    usable = valid_text.any(dim=1)
    if not bool(usable.any()):
        return zero
    return F.smooth_l1_loss(local[usable], target[usable])
