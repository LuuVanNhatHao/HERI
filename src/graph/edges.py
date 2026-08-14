"""Target-independent graph edge construction policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalEdgePairs:
    """Local per-sample edge endpoints before flattening into type stores."""

    samples: torch.Tensor
    src_local: torch.Tensor
    dst_local: torch.Tensor
    strengths: torch.Tensor | None = None
    confidences: torch.Tensor | None = None


def empty_pairs(device: torch.device) -> LocalEdgePairs:
    """Return an empty local edge container."""

    empty = torch.empty(0, dtype=torch.long, device=device)
    return LocalEdgePairs(samples=empty, src_local=empty, dst_local=empty)


def spatial_knn_pairs(
    *,
    src_centers: torch.Tensor,
    src_mask: torch.Tensor,
    dst_centers: torch.Tensor,
    dst_mask: torch.Tensor,
    k: int,
) -> LocalEdgePairs:
    """Connect each valid source node to k nearest valid destination nodes."""

    _validate_batched_mask(src_centers, src_mask, name="src")
    _validate_batched_mask(dst_centers, dst_mask, name="dst")
    samples: list[torch.Tensor] = []
    src_locals: list[torch.Tensor] = []
    dst_locals: list[torch.Tensor] = []
    for batch_index in range(src_mask.shape[0]):
        src_indices = torch.nonzero(src_mask[batch_index], as_tuple=False).flatten()
        dst_indices = torch.nonzero(dst_mask[batch_index], as_tuple=False).flatten()
        if src_indices.numel() == 0 or dst_indices.numel() == 0:
            continue
        distances = torch.cdist(src_centers[batch_index, src_indices], dst_centers[batch_index, dst_indices])
        topk = min(int(k), int(dst_indices.numel()))
        nearest = torch.topk(distances, k=topk, largest=False, dim=1).indices
        src_expanded = src_indices.unsqueeze(1).expand(-1, topk).reshape(-1)
        dst_selected = dst_indices[nearest.reshape(-1)]
        samples.append(torch.full_like(src_expanded, batch_index))
        src_locals.append(src_expanded)
        dst_locals.append(dst_selected)
    return _cat_pairs(samples, src_locals, dst_locals, src_centers.device)


def similarity_topk_pairs(
    *,
    similarity: torch.Tensor,
    src_mask: torch.Tensor,
    dst_mask: torch.Tensor,
    k: int,
    confidence: torch.Tensor | None = None,
) -> LocalEdgePairs:
    """Connect each valid source to top-k valid destinations by similarity."""

    if similarity.ndim != 3:
        raise ValueError("similarity must be [B,S,D]")
    if src_mask.shape != similarity.shape[:2]:
        raise ValueError("src_mask must align with similarity source dimension")
    if dst_mask.shape != (similarity.shape[0], similarity.shape[2]):
        raise ValueError("dst_mask must align with similarity destination dimension")
    samples: list[torch.Tensor] = []
    src_locals: list[torch.Tensor] = []
    dst_locals: list[torch.Tensor] = []
    strengths: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    if confidence is not None and confidence.shape != similarity.shape:
        raise ValueError("confidence must align with similarity [B,S,D]")
    for batch_index in range(similarity.shape[0]):
        src_indices = torch.nonzero(src_mask[batch_index], as_tuple=False).flatten()
        dst_indices = torch.nonzero(dst_mask[batch_index], as_tuple=False).flatten()
        if src_indices.numel() == 0 or dst_indices.numel() == 0:
            continue
        scores = similarity[batch_index][src_indices][:, dst_indices]
        topk = min(int(k), int(dst_indices.numel()))
        selected = torch.topk(scores, k=topk, largest=True, dim=1).indices
        src_expanded = src_indices.unsqueeze(1).expand(-1, topk).reshape(-1)
        dst_selected = dst_indices[selected.reshape(-1)]
        selected_scores = scores.gather(1, selected).reshape(-1)
        samples.append(torch.full_like(src_expanded, batch_index))
        src_locals.append(src_expanded)
        dst_locals.append(dst_selected)
        strengths.append(selected_scores)
        if confidence is not None:
            confidence_scores = confidence[batch_index][src_indices][:, dst_indices]
            confidences.append(confidence_scores.gather(1, selected).reshape(-1))
    pairs = _cat_pairs(samples, src_locals, dst_locals, similarity.device)
    return LocalEdgePairs(
        samples=pairs.samples,
        src_local=pairs.src_local,
        dst_local=pairs.dst_local,
        strengths=torch.cat(strengths) if strengths else similarity.new_zeros((0,)),
        confidences=(torch.cat(confidences) if confidences else None),
    )


def word_aligned_pairs(
    *,
    src_word_ids: torch.Tensor,
    src_mask: torch.Tensor,
    dst_word_ids: torch.Tensor,
    dst_mask: torch.Tensor,
) -> LocalEdgePairs:
    """Connect tokens from different tokenizers only when word IDs match."""

    if src_word_ids.shape != src_mask.shape or dst_word_ids.shape != dst_mask.shape:
        raise ValueError("word IDs must align with their masks")
    if src_mask.shape[0] != dst_mask.shape[0]:
        raise ValueError("source and destination word IDs must share a batch")
    samples: list[torch.Tensor] = []
    src_locals: list[torch.Tensor] = []
    dst_locals: list[torch.Tensor] = []
    for batch_index in range(src_mask.shape[0]):
        src_indices = torch.nonzero(src_mask[batch_index] & (src_word_ids[batch_index] >= 0), as_tuple=False).flatten()
        dst_indices = torch.nonzero(dst_mask[batch_index] & (dst_word_ids[batch_index] >= 0), as_tuple=False).flatten()
        if src_indices.numel() == 0 or dst_indices.numel() == 0:
            continue
        matches = src_word_ids[batch_index, src_indices].unsqueeze(1) == dst_word_ids[
            batch_index, dst_indices
        ].unsqueeze(0)
        coordinates = torch.nonzero(matches, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        src_selected = src_indices[coordinates[:, 0]]
        dst_selected = dst_indices[coordinates[:, 1]]
        samples.append(torch.full_like(src_selected, batch_index))
        src_locals.append(src_selected)
        dst_locals.append(dst_selected)
    return _cat_pairs(samples, src_locals, dst_locals, src_mask.device)


def full_bipartite_pairs(
    *,
    src_mask: torch.Tensor,
    dst_mask: torch.Tensor,
) -> LocalEdgePairs:
    """Connect all valid source nodes to all valid destination nodes per sample."""

    if src_mask.ndim != 2 or dst_mask.ndim != 2 or src_mask.shape[0] != dst_mask.shape[0]:
        raise ValueError("src_mask and dst_mask must be [B,N] with the same batch")
    samples: list[torch.Tensor] = []
    src_locals: list[torch.Tensor] = []
    dst_locals: list[torch.Tensor] = []
    for batch_index in range(src_mask.shape[0]):
        src_indices = torch.nonzero(src_mask[batch_index], as_tuple=False).flatten()
        dst_indices = torch.nonzero(dst_mask[batch_index], as_tuple=False).flatten()
        if src_indices.numel() == 0 or dst_indices.numel() == 0:
            continue
        src_grid = src_indices.repeat_interleave(dst_indices.numel())
        dst_grid = dst_indices.repeat(src_indices.numel())
        samples.append(torch.full_like(src_grid, batch_index))
        src_locals.append(src_grid)
        dst_locals.append(dst_grid)
    return _cat_pairs(samples, src_locals, dst_locals, src_mask.device)


def evidence_to_hypothesis_pairs(
    *,
    evidence_mask: torch.Tensor,
    hypothesis_indices: tuple[int, ...],
) -> LocalEdgePairs:
    """Connect every valid evidence node to the configured hypothesis indices."""

    if evidence_mask.ndim != 2:
        raise ValueError("evidence_mask must be [B,N]")
    samples: list[torch.Tensor] = []
    src_locals: list[torch.Tensor] = []
    dst_locals: list[torch.Tensor] = []
    for batch_index in range(evidence_mask.shape[0]):
        evidence_indices = torch.nonzero(evidence_mask[batch_index], as_tuple=False).flatten()
        if evidence_indices.numel() == 0:
            continue
        hyp = torch.tensor(hypothesis_indices, dtype=torch.long, device=evidence_mask.device)
        src_grid = evidence_indices.repeat_interleave(hyp.numel())
        dst_grid = hyp.repeat(evidence_indices.numel())
        samples.append(torch.full_like(src_grid, batch_index))
        src_locals.append(src_grid)
        dst_locals.append(dst_grid)
    return _cat_pairs(samples, src_locals, dst_locals, evidence_mask.device)


def hypothesis_to_evidence_pairs(
    *,
    evidence_mask: torch.Tensor,
    hypothesis_indices: tuple[int, ...],
) -> LocalEdgePairs:
    """Connect configured hypothesis nodes to every valid evidence node."""

    pairs = evidence_to_hypothesis_pairs(
        evidence_mask=evidence_mask,
        hypothesis_indices=hypothesis_indices,
    )
    return LocalEdgePairs(
        samples=pairs.samples,
        src_local=pairs.dst_local,
        dst_local=pairs.src_local,
    )


def reverse_pairs(pairs: LocalEdgePairs) -> LocalEdgePairs:
    """Reverse source and destination local endpoints."""

    return LocalEdgePairs(
        samples=pairs.samples,
        src_local=pairs.dst_local,
        dst_local=pairs.src_local,
        strengths=pairs.strengths,
        confidences=pairs.confidences,
    )


def _validate_batched_mask(values: torch.Tensor, mask: torch.Tensor, *, name: str) -> None:
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"{name}_centers must be [B,N,2]")
    if mask.shape != values.shape[:2]:
        raise ValueError(f"{name}_mask must align with centers")


def _cat_pairs(
    samples: list[torch.Tensor],
    src_locals: list[torch.Tensor],
    dst_locals: list[torch.Tensor],
    device: torch.device,
) -> LocalEdgePairs:
    if not samples:
        return empty_pairs(device)
    return LocalEdgePairs(
        samples=torch.cat(samples).to(torch.long),
        src_local=torch.cat(src_locals).to(torch.long),
        dst_local=torch.cat(dst_locals).to(torch.long),
    )
