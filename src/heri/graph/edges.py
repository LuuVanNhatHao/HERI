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
        distances = torch.cdist(
            src_centers[batch_index, src_indices].float(),
            dst_centers[batch_index, dst_indices].float(),
        )
        topk = min(int(k), int(dst_indices.numel()))
        nearest = torch.argsort(distances, dim=1, stable=True)[:, :topk]
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
        selected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :topk]
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
        src_indices = torch.nonzero(
            src_mask[batch_index] & (src_word_ids[batch_index] >= 0), as_tuple=False
        ).flatten()
        dst_indices = torch.nonzero(
            dst_mask[batch_index] & (dst_word_ids[batch_index] >= 0), as_tuple=False
        ).flatten()
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


def span_aligned_pairs(
    *,
    src_spans: torch.Tensor,
    src_mask: torch.Tensor,
    dst_spans: torch.Tensor,
    dst_mask: torch.Tensor,
) -> LocalEdgePairs:
    """Cross-tokenizer span overlap; nearest interval fallback for each source."""
    _validate_batched_mask(src_spans, src_mask, name="src")
    _validate_batched_mask(dst_spans, dst_mask, name="dst")
    samples, sources, destinations = [], [], []
    for batch in range(src_mask.shape[0]):
        si = torch.nonzero(
            src_mask[batch] & (src_spans[batch, :, 0] >= 0), as_tuple=False
        ).flatten()
        di = torch.nonzero(
            dst_mask[batch] & (dst_spans[batch, :, 0] >= 0), as_tuple=False
        ).flatten()
        if not si.numel() or not di.numel():
            continue
        left = src_spans[batch, si].float()
        right = dst_spans[batch, di].float()
        overlap = torch.minimum(left[:, None, 1], right[None, :, 1]) > torch.maximum(
            left[:, None, 0], right[None, :, 0]
        )
        distance = (
            torch.maximum(left[:, None, 0], right[None, :, 0])
            - torch.minimum(left[:, None, 1], right[None, :, 1])
        ).clamp_min(0)
        missing = ~overlap.any(dim=1)
        if bool(missing.any()):
            nearest = distance[missing].argmin(dim=1)
            overlap[torch.nonzero(missing, as_tuple=False).flatten(), nearest] = True
        coordinates = overlap.nonzero(as_tuple=False)
        sources.append(si[coordinates[:, 0]])
        destinations.append(di[coordinates[:, 1]])
        samples.append(torch.full_like(sources[-1], batch))
    return _cat_pairs(samples, sources, destinations, src_mask.device)


def intra_spatial_pairs(
    *, centers: torch.Tensor, mask: torch.Tensor, k: int = 4, bidirectional: bool = True
) -> LocalEdgePairs:
    """Within-family spatial kNN, excluding self and duplicate reverse edges."""
    samples, sources, destinations = [], [], []
    for batch in range(mask.shape[0]):
        indices = mask[batch].nonzero(as_tuple=False).flatten()
        count = indices.numel()
        if count < 2:
            continue
        distance = torch.cdist(
            centers[batch, indices].float(), centers[batch, indices].float()
        )
        distance.fill_diagonal_(torch.inf)
        nearest = torch.argsort(distance, dim=1, stable=True)[:, : min(k, count - 1)]
        src = indices[:, None].expand_as(nearest).reshape(-1)
        dst = indices[nearest.reshape(-1)]
        edges = torch.stack([src, dst], dim=1)
        if bidirectional:
            edges = torch.cat([edges, edges.flip(1)]).unique(dim=0)
        sources.append(edges[:, 0])
        destinations.append(edges[:, 1])
        samples.append(torch.full_like(edges[:, 0], batch))
    return _cat_pairs(samples, sources, destinations, mask.device)


def sequential_neighbor_pairs(
    *, source_indices: torch.Tensor, mask: torch.Tensor
) -> LocalEdgePairs:
    """Connect immediate predecessor/successor in retained original token order."""
    samples, sources, destinations = [], [], []
    for batch in range(mask.shape[0]):
        indices = mask[batch].nonzero(as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        indices = indices[torch.argsort(source_indices[batch, indices], stable=True)]
        src = torch.cat([indices[:-1], indices[1:]])
        dst = torch.cat([indices[1:], indices[:-1]])
        samples.append(torch.full_like(src, batch))
        sources.append(src)
        destinations.append(dst)
    return _cat_pairs(samples, sources, destinations, mask.device)


def full_bipartite_pairs(
    *,
    src_mask: torch.Tensor,
    dst_mask: torch.Tensor,
) -> LocalEdgePairs:
    """Connect all valid source nodes to all valid destination nodes per sample."""

    if (
        src_mask.ndim != 2
        or dst_mask.ndim != 2
        or src_mask.shape[0] != dst_mask.shape[0]
    ):
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
        evidence_indices = torch.nonzero(
            evidence_mask[batch_index], as_tuple=False
        ).flatten()
        if evidence_indices.numel() == 0:
            continue
        hyp = torch.tensor(
            hypothesis_indices, dtype=torch.long, device=evidence_mask.device
        )
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


def _validate_batched_mask(
    values: torch.Tensor, mask: torch.Tensor, *, name: str
) -> None:
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
