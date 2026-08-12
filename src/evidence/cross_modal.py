"""Target-independent cross-modal semantic metadata."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .outputs import CrossModalMetadata, EvidenceSet


def cosine_similarity_metadata(
    *,
    visual: EvidenceSet,
    text: EvidenceSet,
) -> CrossModalMetadata:
    """Compute masked cosine similarity between selected SigLIP-derived nodes."""

    if visual.features.shape[-1] != text.features.shape[-1]:
        raise ValueError("Cross-modal cosine requires matched feature dimensions")
    visual_norm = F.normalize(visual.features, p=2, dim=-1, eps=1e-12)
    text_norm = F.normalize(text.features, p=2, dim=-1, eps=1e-12)
    similarity = torch.matmul(visual_norm, text_norm.transpose(1, 2))
    valid = visual.mask.unsqueeze(2) & text.mask.unsqueeze(1)
    similarity = similarity.masked_fill(~valid, 0.0)
    return CrossModalMetadata(
        similarity=similarity,
        visual_mask=visual.mask,
        text_mask=text.mask,
        visual_source_indices=visual.source_indices,
        text_source_indices=text.source_indices,
        method="cosine",
    )
