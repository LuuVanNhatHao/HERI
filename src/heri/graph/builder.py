"""Target-independent graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from heri.evidence.outputs import EvidenceSet, HeterogeneousEvidenceBatch

from .edges import (
    LocalEdgePairs,
    evidence_to_hypothesis_pairs,
    reverse_pairs,
    similarity_topk_pairs,
    spatial_knn_pairs,
    word_aligned_pairs,
    span_aligned_pairs,
    intra_spatial_pairs,
    sequential_neighbor_pairs,
)
from .outputs import GraphBatch
from .schema import (
    HYPOTHESIS,
    SEMANTIC_ALIGNS,
    SPATIAL_CORRESPONDS,
    SPATIAL_NEIGHBOR,
    SEQUENTIAL_NEIGHBOR,
    TEXT_ALIGNMENT,
    TEXT_CONTEXT,
    TOKEN_CONTEXTUALIZES,
    VISUAL_FORENSIC,
    VISUAL_SEMANTIC,
    HypothesisSpec,
    edge_types_for_hypotheses,
    support_relation_name,
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
    """Build typed graph nodes and edges from evidence only."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.hidden_dim = int(config["hidden_dim"])
        self.edge_config = config["edges"]
        self.hypothesis_spec = HypothesisSpec.from_mapping(config.get("hypotheses"))
        self.hypothesis_to_index = {
            name: index for index, name in enumerate(self.hypothesis_spec.order)
        }
        self.edge_types = edge_types_for_hypotheses(self.hypothesis_spec.order)
        _validate_edge_config(self.edge_config)
        self.feedback_enabled = bool(config["reasoning"].get("hypothesis_feedback", True))
        self.include_hypothesis_nodes = bool(config.get("include_hypothesis_nodes", True))
        self.pooled = config["reasoning"].get("operator", "graph") == "pooled"

    def build(
        self,
        evidence: HeterogeneousEvidenceBatch,
        hypothesis_features: torch.Tensor,
        *,
        support_hypothesis_features: torch.Tensor | None = None,
    ) -> GraphBatch:
        """Return a flattened heterogeneous graph batch with no target inputs."""

        evidence.validate()
        if hypothesis_features.ndim != 3 or hypothesis_features.shape[1] != len(self.hypothesis_spec.order):
            raise ValueError(
                f"hypothesis_features must be [B,{len(self.hypothesis_spec.order)},Dg]"
            )
        if hypothesis_features.shape[-1] != self.hidden_dim:
            raise ValueError("hypothesis_features last dimension must equal graph.hidden_dim")
        batch_size = hypothesis_features.shape[0]
        if batch_size != evidence.visual_semantic.features.shape[0]:
            raise ValueError("hypothesis_features and evidence must share the batch size")
        stores = {
            VISUAL_SEMANTIC: _evidence_store(evidence.visual_semantic, hidden_dim=self.hidden_dim),
            VISUAL_FORENSIC: _evidence_store(evidence.visual_forensic, hidden_dim=self.hidden_dim),
            TEXT_ALIGNMENT: _evidence_store(evidence.text_alignment, hidden_dim=self.hidden_dim),
            TEXT_CONTEXT: _evidence_store(evidence.text_context, hidden_dim=self.hidden_dim),
            HYPOTHESIS: _hypothesis_store(hypothesis_features if self.include_hypothesis_nodes and not self.pooled
                                          else hypothesis_features[:, :0]),
        }
        edge_index_dict, edge_sample_indices, edge_attr_dict = self._build_edges(
            evidence=evidence,
            stores=stores,
            hypothesis_features=(hypothesis_features if support_hypothesis_features is None
                                 else support_hypothesis_features),
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
                "evidence_mode": str(self.config.get("evidence_mode", "heterogeneous")),
                "hypothesis_order": list(self.hypothesis_spec.order),
                "node_types": list(stores),
                "edge_types": list(self.edge_types),
                "include_hypothesis_nodes": self.include_hypothesis_nodes and not self.pooled,
                "reasoning_operator": "pooled" if self.pooled else "graph",
            },
        )
        graph.validate()
        return graph

    def _build_edges(
        self,
        *,
        evidence: HeterogeneousEvidenceBatch,
        stores: Mapping[str, _NodeStore],
        hypothesis_features: torch.Tensor,
    ) -> tuple[
        dict[EdgeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
    ]:
        device = evidence.visual_semantic.features.device
        edge_index_dict = {
            edge_type: torch.empty((2, 0), dtype=torch.long, device=device)
            for edge_type in self.edge_types
        }
        edge_sample_indices = {
            edge_type: torch.empty(0, dtype=torch.long, device=device)
            for edge_type in self.edge_types
        }
        edge_attr_dict: dict[EdgeType, torch.Tensor] = {}

        if self.pooled:
            return edge_index_dict, edge_sample_indices, {
                edge: torch.empty((0,2), device=device, dtype=torch.float32) for edge in self.edge_types
            }

        for node_type, evidence_set in evidence.families().items():
            if node_type in (VISUAL_SEMANTIC, VISUAL_FORENSIC):
                cfg = self.edge_config.get("intra_visual", {})
                intra = intra_spatial_pairs(centers=_require_centers(evidence_set), mask=evidence_set.mask,
                                           k=int(cfg.get("k",4)), bidirectional=bool(cfg.get("bidirectional",True)))
                edge_type = (node_type, SPATIAL_NEIGHBOR, node_type)
            else:
                intra = sequential_neighbor_pairs(source_indices=evidence_set.source_indices, mask=evidence_set.mask)
                edge_type = (node_type, SEQUENTIAL_NEIGHBOR, node_type)
            self._add_pairs(edge_index_dict, edge_sample_indices, stores, edge_type=edge_type, pairs=intra)

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
            raise ValueError("visual_semantic_text_alignment requires cross-modal similarity metadata")
        cross_modal_enabled = bool(self.edge_config.get("cross_modal", {}).get("enabled", True))
        semantic_confidence = evidence.cross_modal_metadata.edge_confidence
        if semantic_confidence is None:
            visual_rho = evidence.visual_semantic.reliability_scores
            text_rho = evidence.text_alignment.reliability_scores
            if visual_rho is None:
                visual_rho = evidence.visual_semantic.mask.float()
            if text_rho is None:
                text_rho = evidence.text_alignment.mask.float()
            valid_semantic = evidence.visual_semantic.mask.unsqueeze(2) & evidence.text_alignment.mask.unsqueeze(1)
            semantic_confidence = (
                ((1.0 + evidence.cross_modal_metadata.similarity.float()) * .5).clamp(0.0,1.0)
                * torch.sqrt((visual_rho.float().unsqueeze(2)*text_rho.float().unsqueeze(1)).clamp_min(1e-16))
            ).clamp_min(1e-8).masked_fill(~valid_semantic,0.0)
        pairs = similarity_topk_pairs(
            similarity=evidence.cross_modal_metadata.similarity,
            src_mask=evidence.visual_semantic.mask,
            dst_mask=evidence.text_alignment.mask,
            k=int(vs_ta["k"]) if cross_modal_enabled else 0,
            confidence=semantic_confidence,
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
        if ta_tc["method"] == "span_aligned":
            pairs = span_aligned_pairs(
                src_spans=_require_spans(evidence.text_alignment), src_mask=evidence.text_alignment.mask,
                dst_spans=_require_spans(evidence.text_context), dst_mask=evidence.text_context.mask,
            )
        else:
            if evidence.text_alignment.word_ids is None or evidence.text_context.word_ids is None:
                raise ValueError("word-aligned text edges require word_ids for both tokenizer families")
            pairs = word_aligned_pairs(
                src_word_ids=evidence.text_alignment.word_ids, src_mask=evidence.text_alignment.mask,
                dst_word_ids=evidence.text_context.word_ids, dst_mask=evidence.text_context.mask,
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

        if self.include_hypothesis_nodes:
            self._add_evidence_support_edges(
                edge_index_dict,
                edge_sample_indices,
                edge_attr_dict,
                evidence,
                stores,
                hypothesis_features,
            )
        if self.include_hypothesis_nodes and self.feedback_enabled and bool(self.edge_config.get("feedback", {}).get("enabled", True)):
            self._add_feedback_edges(
                edge_index_dict,
                edge_sample_indices,
                edge_attr_dict,
                evidence,
                stores,
                hypothesis_features,
            )
        for edge_type in self.edge_types:
            if edge_type not in edge_attr_dict:
                edge_count = edge_index_dict[edge_type].shape[1]
                attrs = torch.zeros((edge_count, 2), device=device, dtype=torch.float32)
                if edge_count:
                    attrs[:, 0] = 1.0
                    src_type, _, dst_type = edge_type
                    src, dst = edge_index_dict[edge_type]
                    src_store, dst_store = stores[src_type], stores[dst_type]
                    source = evidence.families()[src_type]
                    target = evidence.families()[dst_type]
                    sr = (torch.ones(edge_count,device=device) if source.reliability_scores is None else
                          source.reliability_scores[src_store.sample_indices[src],src_store.local_indices[src]].float())
                    dr = (torch.ones(edge_count,device=device) if target.reliability_scores is None else
                          target.reliability_scores[dst_store.sample_indices[dst],dst_store.local_indices[dst]].float())
                    attrs[:, 1] = torch.sqrt((sr*dr).clamp_min(1e-16)).clamp_min(1e-8)
                edge_attr_dict[edge_type] = attrs
        return edge_index_dict, edge_sample_indices, edge_attr_dict

    def _add_evidence_support_edges(
        self,
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_sample_indices: dict[EdgeType, torch.Tensor],
        edge_attr_dict: dict[EdgeType, torch.Tensor],
        evidence: HeterogeneousEvidenceBatch,
        stores: Mapping[str, _NodeStore],
        hypothesis_features: torch.Tensor,
    ) -> None:
        for source_type, evidence_set, hypotheses in (
            (VISUAL_FORENSIC, evidence.visual_forensic, self.hypothesis_spec.image),
            (TEXT_CONTEXT, evidence.text_context, self.hypothesis_spec.text),
        ):
            for hypothesis in hypotheses:
                relation = _support_relation(hypothesis)
                pairs = _hypothesis_conditioned_pairs(
                    evidence_set=evidence_set,
                    hypothesis_features=hypothesis_features,
                    hypothesis_indices=(self.hypothesis_to_index[hypothesis],),
                )
                edge_type = (source_type, relation, HYPOTHESIS)
                self._add_pairs(
                    edge_index_dict,
                    edge_sample_indices,
                    stores,
                    edge_type=edge_type,
                    pairs=pairs,
                )
                edge_attr_dict[edge_type] = _pair_attributes(pairs, stores, edge_type)

        semantic_config = self.edge_config.get("semantic_to_hypotheses", {})
        if not bool(semantic_config.get("enabled", True)):
            return
        for source_type, evidence_set, hypotheses in (
            (VISUAL_SEMANTIC, evidence.visual_semantic, self.hypothesis_spec.image),
            (TEXT_ALIGNMENT, evidence.text_alignment, self.hypothesis_spec.text),
        ):
            for hypothesis in hypotheses:
                relation = _support_relation(hypothesis)
                pairs = _hypothesis_conditioned_pairs(
                    evidence_set=evidence_set,
                    hypothesis_features=hypothesis_features,
                    hypothesis_indices=(self.hypothesis_to_index[hypothesis],),
                )
                edge_type = (source_type, relation, HYPOTHESIS)
                self._add_pairs(
                    edge_index_dict,
                    edge_sample_indices,
                    stores,
                    edge_type=edge_type,
                    pairs=pairs,
                )
                edge_attr_dict[edge_type] = _pair_attributes(pairs, stores, edge_type)

    def _add_feedback_edges(
        self,
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_sample_indices: dict[EdgeType, torch.Tensor],
        edge_attr_dict: dict[EdgeType, torch.Tensor],
        evidence: HeterogeneousEvidenceBatch,
        stores: Mapping[str, _NodeStore],
        hypothesis_features: torch.Tensor,
    ) -> None:
        feedback_config = self.edge_config.get("feedback", {})
        for edge_type, evidence_set, hypotheses in (
            ((HYPOTHESIS, QUERIES_VISUAL_FORENSIC, VISUAL_FORENSIC), evidence.visual_forensic, self.hypothesis_spec.image),
            ((HYPOTHESIS, QUERIES_TEXT_CONTEXT, TEXT_CONTEXT), evidence.text_context, self.hypothesis_spec.text),
        ):
            support_pairs = _hypothesis_conditioned_pairs(
                evidence_set=evidence_set,
                hypothesis_features=hypothesis_features,
                hypothesis_indices=tuple(self.hypothesis_to_index[name] for name in hypotheses),
            )
            pairs = reverse_pairs(support_pairs)
            self._add_pairs(edge_index_dict, edge_sample_indices, stores, edge_type=edge_type, pairs=pairs)
            edge_attr_dict[edge_type] = _pair_attributes(pairs, stores, edge_type)

        if bool(feedback_config.get("semantic_feedback", True)):
            for edge_type, evidence_set, hypotheses in (
                ((HYPOTHESIS, QUERIES_VISUAL_SEMANTIC, VISUAL_SEMANTIC), evidence.visual_semantic, self.hypothesis_spec.image),
                ((HYPOTHESIS, QUERIES_TEXT_ALIGNMENT, TEXT_ALIGNMENT), evidence.text_alignment, self.hypothesis_spec.text),
            ):
                support_pairs = _hypothesis_conditioned_pairs(
                    evidence_set=evidence_set,
                    hypothesis_features=hypothesis_features,
                    hypothesis_indices=tuple(self.hypothesis_to_index[name] for name in hypotheses),
                )
                pairs = reverse_pairs(support_pairs)
                self._add_pairs(edge_index_dict, edge_sample_indices, stores, edge_type=edge_type, pairs=pairs)
                edge_attr_dict[edge_type] = _pair_attributes(pairs, stores, edge_type)

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
    return support_relation_name(hypothesis)


def _require_centers(evidence_set: EvidenceSet) -> torch.Tensor:
    if evidence_set.normalized_centers is None:
        if evidence_set.features.shape[1] == 0:
            return evidence_set.features.new_zeros((*evidence_set.mask.shape,2))
        raise ValueError(f"{evidence_set.family} normalized_centers are required for spatial graph edges")
    return evidence_set.normalized_centers


def _require_spans(evidence_set: EvidenceSet) -> torch.Tensor:
    if evidence_set.token_spans is not None:
        return evidence_set.token_spans
    # Structural compatibility for synthetic/legacy feature fixtures. Production
    # fast-tokenizer paths provide actual shared-frame character offsets.
    if evidence_set.word_ids is not None:
        ids=evidence_set.word_ids
        return torch.stack([ids,ids+1],dim=-1).masked_fill((ids<0).unsqueeze(-1),-1)
    return torch.full((*evidence_set.mask.shape,2),-1,device=evidence_set.mask.device,dtype=torch.long)


def _validate_edge_config(edge_config: Mapping[str, Any]) -> None:
    allowed_keys = {
        "visual_semantic_visual_forensic",
        "visual_semantic_text_alignment",
        "text_alignment_text_context",
        "semantic_to_hypotheses",
        "feedback",
        "intra_visual", "intra_text", "cross_modal",
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
        if actual_method != expected_method and not (key == "text_alignment_text_context" and actual_method == "span_aligned"):
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
        attrs = stores[src_type].features.new_zeros((int(valid.sum().item()), 2))
        if attrs.numel():
            attrs[:, 1] = 1.0
        return attrs
    confidence = (
        torch.ones_like(pairs.strengths)
        if pairs.confidences is None
        else pairs.confidences
    )
    return torch.stack([pairs.strengths[valid], confidence[valid]], dim=-1).float()


def _hypothesis_conditioned_pairs(
    *,
    evidence_set: EvidenceSet,
    hypothesis_features: torch.Tensor,
    hypothesis_indices: tuple[int, ...],
) -> LocalEdgePairs:
    """Build support edges whose strength and confidence reflect current graph state."""

    pairs = evidence_to_hypothesis_pairs(
        evidence_mask=evidence_set.mask,
        hypothesis_indices=hypothesis_indices,
    )
    if pairs.samples.numel() == 0:
        empty = evidence_set.features.new_zeros((0,))
        return LocalEdgePairs(
            samples=pairs.samples,
            src_local=pairs.src_local,
            dst_local=pairs.dst_local,
            strengths=empty,
            confidences=empty,
        )
    evidence_features = evidence_set.features[pairs.samples, pairs.src_local].float()
    hypotheses = hypothesis_features[pairs.samples, pairs.dst_local].float()
    strengths = torch.nn.functional.cosine_similarity(evidence_features, hypotheses, dim=-1)
    if evidence_set.reliability_scores is None:
        confidences = torch.ones_like(strengths)
    else:
        confidences = evidence_set.reliability_scores[pairs.samples, pairs.src_local].float()
    confidences = (confidences * ((1.0 + strengths) * 0.5).clamp(0.0,1.0)).clamp_min(1e-8)
    return LocalEdgePairs(
        samples=pairs.samples,
        src_local=pairs.src_local,
        dst_local=pairs.dst_local,
        strengths=strengths.to(evidence_set.features.dtype),
        confidences=confidences.to(evidence_set.features.dtype).clamp(0.0, 1.0),
    )
