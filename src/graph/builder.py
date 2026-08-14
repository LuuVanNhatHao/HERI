"""Target-independent Phase-4 graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.evidence.outputs import EvidenceSet, HeterogeneousEvidenceBatch

from .edges import (
    LocalEdgePairs,
    evidence_to_hypothesis_pairs,
    hypothesis_to_evidence_pairs,
    reverse_pairs,
    similarity_topk_pairs,
    spatial_knn_pairs,
    word_aligned_pairs,
)
from .outputs import GraphBatch
from .schema import (
    FACE,
    HYPOTHESIS,
    HYPOTHESIS_TO_INDEX,
    SEMANTIC_ALIGNS,
    SPATIAL_CORRESPONDS,
    SUPPORTS_FA,
    SUPPORTS_FS,
    SUPPORTS_TA,
    SUPPORTS_TS,
    TEXT_ALIGNMENT,
    TEXT_CONTEXT,
    TOKEN_CONTEXTUALIZES,
    VISUAL_FORENSIC,
    VISUAL_SEMANTIC,
    CANONICAL_EDGE_TYPES,
    QUERIES_FACE,
    QUERIES_TEXT_ALIGNMENT,
    QUERIES_TEXT_CONTEXT,
    QUERIES_VISUAL_FORENSIC,
    QUERIES_VISUAL_SEMANTIC,
    EdgeType,
)


@dataclass
class _NodeStore:
    features: torch.Tensor
    sample_indices: torch.Tensor
    local_indices: torch.Tensor
    lookup: torch.Tensor


class HypothesisEvidenceGraphBuilder:
    """Build typed graph nodes and edges from Phase-3 evidence only."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.hidden_dim = int(config["hidden_dim"])
        self.edge_config = config["edges"]
        _validate_edge_config(self.edge_config)
        self.feedback_enabled = bool(config["reasoning"].get("hypothesis_feedback", True))

    def build(
        self,
        evidence: HeterogeneousEvidenceBatch,
        hypothesis_features: torch.Tensor,
    ) -> GraphBatch:
        """Return a flattened heterogeneous graph batch with no target inputs."""

        evidence.validate()
        if hypothesis_features.ndim != 3 or hypothesis_features.shape[1] != 4:
            raise ValueError("hypothesis_features must be [B,4,Dg]")
        if hypothesis_features.shape[-1] != self.hidden_dim:
            raise ValueError("hypothesis_features last dimension must equal graph.hidden_dim")
        batch_size = hypothesis_features.shape[0]
        stores = {
            VISUAL_SEMANTIC: _evidence_store(evidence.visual_semantic, hidden_dim=self.hidden_dim),
            VISUAL_FORENSIC: _evidence_store(evidence.visual_forensic, hidden_dim=self.hidden_dim),
            FACE: _evidence_store(evidence.face, hidden_dim=self.hidden_dim),
            TEXT_ALIGNMENT: _evidence_store(evidence.text_alignment, hidden_dim=self.hidden_dim),
            TEXT_CONTEXT: _evidence_store(evidence.text_context, hidden_dim=self.hidden_dim),
            HYPOTHESIS: _hypothesis_store(hypothesis_features),
        }
        edge_index_dict, edge_sample_indices, edge_attr_dict = self._build_edges(
            evidence=evidence, stores=stores
        )
        graph = GraphBatch(
            x_dict={node_type: store.features for node_type, store in stores.items()},
            edge_index_dict=edge_index_dict,
            node_sample_indices={node_type: store.sample_indices for node_type, store in stores.items()},
            node_local_indices={node_type: store.local_indices for node_type, store in stores.items()},
            edge_sample_indices=edge_sample_indices,
            edge_attr_dict=edge_attr_dict,
            batch_size=batch_size,
            hidden_dim=self.hidden_dim,
            metadata={
                "reasoning_cycles": int(self.config["reasoning"]["cycles"]),
                "hypothesis_feedback": self.feedback_enabled,
            },
        )
        graph.validate()
        return graph

    def _build_edges(
        self,
        *,
        evidence: HeterogeneousEvidenceBatch,
        stores: Mapping[str, _NodeStore],
    ) -> tuple[
        dict[EdgeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
    ]:
        device = evidence.visual_semantic.features.device
        edge_index_dict = {
            edge_type: torch.empty((2, 0), dtype=torch.long, device=device)
            for edge_type in CANONICAL_EDGE_TYPES
        }
        edge_sample_indices = {
            edge_type: torch.empty(0, dtype=torch.long, device=device)
            for edge_type in CANONICAL_EDGE_TYPES
        }
        edge_attr_dict: dict[EdgeType, torch.Tensor] = {}

        vs_vf = self.edge_config["visual_semantic_visual_forensic"]
        pairs = spatial_knn_pairs(
            src_centers=_require_centers(evidence.visual_semantic),
            src_mask=evidence.visual_semantic.mask,
            dst_centers=_require_centers(evidence.visual_forensic),
            dst_mask=evidence.visual_forensic.mask,
            k=int(vs_vf["k"]),
        )
        self._add_pairs(
            edge_index_dict,
            edge_sample_indices,
            stores,
            edge_type=(VISUAL_SEMANTIC, SPATIAL_CORRESPONDS, VISUAL_FORENSIC),
            pairs=pairs,
        )
        if bool(vs_vf.get("bidirectional", True)):
            self._add_pairs(
                edge_index_dict,
                edge_sample_indices,
                stores,
                edge_type=(VISUAL_FORENSIC, SPATIAL_CORRESPONDS, VISUAL_SEMANTIC),
                pairs=reverse_pairs(pairs),
            )

        vs_ta = self.edge_config["visual_semantic_text_alignment"]
        if evidence.cross_modal_metadata.similarity is None:
            raise ValueError("visual_semantic_text_alignment requires Phase-3 cross-modal similarity metadata")
        pairs = similarity_topk_pairs(
            similarity=evidence.cross_modal_metadata.similarity,
            src_mask=evidence.visual_semantic.mask,
            dst_mask=evidence.text_alignment.mask,
            k=int(vs_ta["k"]),
            confidence=evidence.cross_modal_metadata.edge_confidence,
        )
        forward_semantic_type = (VISUAL_SEMANTIC, SEMANTIC_ALIGNS, TEXT_ALIGNMENT)
        self._add_pairs(
            edge_index_dict,
            edge_sample_indices,
            stores,
            edge_type=forward_semantic_type,
            pairs=pairs,
        )
        edge_attr_dict[forward_semantic_type] = _pair_attributes(pairs, stores, forward_semantic_type)
        if bool(vs_ta.get("bidirectional", True)):
            reverse_semantic_type = (TEXT_ALIGNMENT, SEMANTIC_ALIGNS, VISUAL_SEMANTIC)
            reversed_pairs = reverse_pairs(pairs)
            self._add_pairs(
                edge_index_dict,
                edge_sample_indices,
                stores,
                edge_type=reverse_semantic_type,
                pairs=reversed_pairs,
            )
            edge_attr_dict[reverse_semantic_type] = _pair_attributes(
                reversed_pairs, stores, reverse_semantic_type
            )

        ta_tc = self.edge_config["text_alignment_text_context"]
        if evidence.text_alignment.word_ids is None or evidence.text_context.word_ids is None:
            raise ValueError("word-aligned text edges require word_ids for both tokenizer families")
        pairs = word_aligned_pairs(
            src_word_ids=evidence.text_alignment.word_ids,
            src_mask=evidence.text_alignment.mask,
            dst_word_ids=evidence.text_context.word_ids,
            dst_mask=evidence.text_context.mask,
        )
        self._add_pairs(
            edge_index_dict,
            edge_sample_indices,
            stores,
            edge_type=(TEXT_ALIGNMENT, TOKEN_CONTEXTUALIZES, TEXT_CONTEXT),
            pairs=pairs,
        )
        if bool(ta_tc.get("bidirectional", True)):
            self._add_pairs(
                edge_index_dict,
                edge_sample_indices,
                stores,
                edge_type=(TEXT_CONTEXT, TOKEN_CONTEXTUALIZES, TEXT_ALIGNMENT),
                pairs=reverse_pairs(pairs),
            )

        self._add_evidence_support_edges(edge_index_dict, edge_sample_indices, evidence, stores)
        if self.feedback_enabled and bool(self.edge_config.get("feedback", {}).get("enabled", True)):
            self._add_feedback_edges(edge_index_dict, edge_sample_indices, evidence, stores)
        for edge_type in CANONICAL_EDGE_TYPES:
            edge_attr_dict.setdefault(
                edge_type,
                evidence.visual_semantic.features.new_zeros(
                    (edge_index_dict[edge_type].shape[1], 2)
                ),
            )
        return edge_index_dict, edge_sample_indices, edge_attr_dict

    def _add_evidence_support_edges(
        self,
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_sample_indices: dict[EdgeType, torch.Tensor],
        evidence: HeterogeneousEvidenceBatch,
        stores: Mapping[str, _NodeStore],
    ) -> None:
        for source_type, evidence_set, hypotheses in (
            (VISUAL_FORENSIC, evidence.visual_forensic, ("FS", "FA")),
            (FACE, evidence.face, ("FS", "FA")),
            (TEXT_CONTEXT, evidence.text_context, ("TS", "TA")),
        ):
            for hypothesis in hypotheses:
                relation = _support_relation(hypothesis)
                pairs = evidence_to_hypothesis_pairs(
                    evidence_mask=evidence_set.mask,
                    hypothesis_indices=(HYPOTHESIS_TO_INDEX[hypothesis],),
                )
                self._add_pairs(
                    edge_index_dict,
                    edge_sample_indices,
                    stores,
                    edge_type=(source_type, relation, HYPOTHESIS),
                    pairs=pairs,
                )

        semantic_config = self.edge_config.get("semantic_to_hypotheses", {})
        if not bool(semantic_config.get("enabled", True)):
            return
        for source_type, evidence_set in (
            (VISUAL_SEMANTIC, evidence.visual_semantic),
            (TEXT_ALIGNMENT, evidence.text_alignment),
        ):
            for hypothesis in ("FS", "FA", "TS", "TA"):
                relation = _support_relation(hypothesis)
                pairs = evidence_to_hypothesis_pairs(
                    evidence_mask=evidence_set.mask,
                    hypothesis_indices=(HYPOTHESIS_TO_INDEX[hypothesis],),
                )
                self._add_pairs(
                    edge_index_dict,
                    edge_sample_indices,
                    stores,
                    edge_type=(source_type, relation, HYPOTHESIS),
                    pairs=pairs,
                )

    def _add_feedback_edges(
        self,
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_sample_indices: dict[EdgeType, torch.Tensor],
        evidence: HeterogeneousEvidenceBatch,
        stores: Mapping[str, _NodeStore],
    ) -> None:
        feedback_config = self.edge_config.get("feedback", {})
        for edge_type, evidence_mask, hypotheses in (
            ((HYPOTHESIS, QUERIES_FACE, FACE), evidence.face.mask, ("FS", "FA")),
            ((HYPOTHESIS, QUERIES_VISUAL_FORENSIC, VISUAL_FORENSIC), evidence.visual_forensic.mask, ("FS", "FA")),
            ((HYPOTHESIS, QUERIES_TEXT_CONTEXT, TEXT_CONTEXT), evidence.text_context.mask, ("TS", "TA")),
        ):
            pairs = hypothesis_to_evidence_pairs(
                evidence_mask=evidence_mask,
                hypothesis_indices=tuple(HYPOTHESIS_TO_INDEX[name] for name in hypotheses),
            )
            self._add_pairs(edge_index_dict, edge_sample_indices, stores, edge_type=edge_type, pairs=pairs)

        if bool(feedback_config.get("semantic_feedback", False)):
            for edge_type, evidence_mask in (
                ((HYPOTHESIS, QUERIES_VISUAL_SEMANTIC, VISUAL_SEMANTIC), evidence.visual_semantic.mask),
                ((HYPOTHESIS, QUERIES_TEXT_ALIGNMENT, TEXT_ALIGNMENT), evidence.text_alignment.mask),
            ):
                pairs = hypothesis_to_evidence_pairs(
                    evidence_mask=evidence_mask,
                    hypothesis_indices=(0, 1, 2, 3),
                )
                self._add_pairs(edge_index_dict, edge_sample_indices, stores, edge_type=edge_type, pairs=pairs)

    def _add_pairs(
        self,
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_sample_indices: dict[EdgeType, torch.Tensor],
        stores: Mapping[str, _NodeStore],
        *,
        edge_type: EdgeType,
        pairs: LocalEdgePairs,
    ) -> None:
        src_type, _, dst_type = edge_type
        src_lookup = stores[src_type].lookup
        dst_lookup = stores[dst_type].lookup
        if pairs.samples.numel() == 0:
            return
        src_global = src_lookup[pairs.samples, pairs.src_local]
        dst_global = dst_lookup[pairs.samples, pairs.dst_local]
        valid = (src_global >= 0) & (dst_global >= 0)
        if not bool(valid.any()):
            return
        edge_index = torch.stack([src_global[valid], dst_global[valid]], dim=0).to(torch.long)
        edge_index_dict[edge_type] = torch.cat([edge_index_dict[edge_type], edge_index], dim=1)
        edge_sample_indices[edge_type] = torch.cat([edge_sample_indices[edge_type], pairs.samples[valid]], dim=0)


def _evidence_store(evidence_set: EvidenceSet, *, hidden_dim: int) -> _NodeStore:
    if evidence_set.features.shape[-1] != hidden_dim:
        raise ValueError(
            f"{evidence_set.family} feature dim {evidence_set.features.shape[-1]} "
            f"does not match graph.hidden_dim {hidden_dim}"
        )
    mask = evidence_set.mask
    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.numel() == 0:
        features = evidence_set.features.new_zeros((0, hidden_dim))
        sample_indices = torch.empty(0, dtype=torch.long, device=evidence_set.features.device)
        local_indices = torch.empty(0, dtype=torch.long, device=evidence_set.features.device)
    else:
        sample_indices = coordinates[:, 0].to(torch.long)
        local_indices = coordinates[:, 1].to(torch.long)
        features = evidence_set.features[mask]
    lookup = torch.full(mask.shape, -1, dtype=torch.long, device=evidence_set.features.device)
    if sample_indices.numel():
        lookup[sample_indices, local_indices] = torch.arange(
            sample_indices.numel(),
            dtype=torch.long,
            device=evidence_set.features.device,
        )
    return _NodeStore(
        features=features,
        sample_indices=sample_indices,
        local_indices=local_indices,
        lookup=lookup,
    )


def _hypothesis_store(hypothesis_features: torch.Tensor) -> _NodeStore:
    batch_size, hypothesis_count, hidden_dim = hypothesis_features.shape
    features = hypothesis_features.reshape(batch_size * hypothesis_count, hidden_dim)
    device = hypothesis_features.device
    sample_indices = torch.arange(batch_size, device=device, dtype=torch.long).repeat_interleave(hypothesis_count)
    local_indices = torch.arange(hypothesis_count, device=device, dtype=torch.long).repeat(batch_size)
    lookup = torch.arange(batch_size * hypothesis_count, device=device, dtype=torch.long).reshape(
        batch_size,
        hypothesis_count,
    )
    return _NodeStore(
        features=features,
        sample_indices=sample_indices,
        local_indices=local_indices,
        lookup=lookup,
    )


def _support_relation(hypothesis: str) -> str:
    return {
        "FS": SUPPORTS_FS,
        "FA": SUPPORTS_FA,
        "TS": SUPPORTS_TS,
        "TA": SUPPORTS_TA,
    }[hypothesis]


def _require_centers(evidence_set: EvidenceSet) -> torch.Tensor:
    if evidence_set.normalized_centers is None:
        raise ValueError(f"{evidence_set.family} normalized_centers are required for spatial graph edges")
    return evidence_set.normalized_centers


def _validate_edge_config(edge_config: Mapping[str, Any]) -> None:
    allowed_keys = {
        "visual_semantic_visual_forensic",
        "visual_semantic_text_alignment",
        "text_alignment_text_context",
        "semantic_to_hypotheses",
        "feedback",
    }
    unknown = sorted(set(edge_config) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown graph edge config keys: {unknown}")
    expected_methods = {
        "visual_semantic_visual_forensic": "spatial_knn",
        "visual_semantic_text_alignment": "similarity_topk",
        "text_alignment_text_context": "word_aligned",
    }
    for key, expected_method in expected_methods.items():
        if key not in edge_config:
            raise ValueError(f"Missing graph edge config: {key}")
        actual_method = str(edge_config[key].get("method"))
        if actual_method != expected_method:
            raise ValueError(f"{key}.method must be {expected_method}, got {actual_method!r}")
    for key in ("visual_semantic_visual_forensic", "visual_semantic_text_alignment"):
        if int(edge_config[key].get("k", 0)) <= 0:
            raise ValueError(f"{key}.k must be positive")


def _pair_attributes(
    pairs: LocalEdgePairs,
    stores: Mapping[str, _NodeStore],
    edge_type: EdgeType,
) -> torch.Tensor:
    """Flatten continuous similarity/confidence in the same order as edge indices."""

    src_type, _, dst_type = edge_type
    src_global = stores[src_type].lookup[pairs.samples, pairs.src_local]
    dst_global = stores[dst_type].lookup[pairs.samples, pairs.dst_local]
    valid = (src_global >= 0) & (dst_global >= 0)
    if pairs.strengths is None:
        return stores[src_type].features.new_zeros((int(valid.sum().item()), 2))
    confidence = (
        torch.ones_like(pairs.strengths)
        if pairs.confidences is None
        else pairs.confidences
    )
    return torch.stack([pairs.strengths[valid], confidence[valid]], dim=-1).to(
        dtype=stores[src_type].features.dtype
    )
