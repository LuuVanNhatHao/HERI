"""Typed Phase-4 graph output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.evidence.outputs import EvidenceSet

from .schema import CANONICAL_EDGE_TYPES, HYPOTHESIS, HYPOTHESIS_ORDER, NODE_TYPES, EdgeType


@dataclass
class GraphBatch:
    """A PyG-ready heterogeneous graph batch with explicit sample boundaries."""

    x_dict: dict[str, torch.Tensor]
    edge_index_dict: dict[EdgeType, torch.Tensor]
    node_sample_indices: dict[str, torch.Tensor]
    node_local_indices: dict[str, torch.Tensor]
    edge_sample_indices: dict[EdgeType, torch.Tensor]
    edge_attr_dict: dict[EdgeType, torch.Tensor]
    batch_size: int
    hidden_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate node tensors, edge indices, and cross-sample isolation."""

        for node_type in NODE_TYPES:
            if node_type not in self.x_dict:
                raise ValueError(f"Missing node type in graph batch: {node_type}")
            features = self.x_dict[node_type]
            if features.ndim != 2 or features.shape[-1] != self.hidden_dim:
                raise ValueError(f"{node_type} features must be [N,{self.hidden_dim}], got {tuple(features.shape)}")
            if not torch.isfinite(features).all():
                raise FloatingPointError(f"{node_type} graph features contain non-finite values")
            sample_indices = self.node_sample_indices[node_type]
            local_indices = self.node_local_indices[node_type]
            if sample_indices.shape != (features.shape[0],) or local_indices.shape != (features.shape[0],):
                raise ValueError(f"{node_type} node index metadata shape mismatch")
            if sample_indices.numel() and ((sample_indices < 0).any() or (sample_indices >= self.batch_size).any()):
                raise ValueError(f"{node_type} node sample index out of range")

        for edge_type in CANONICAL_EDGE_TYPES:
            if edge_type not in self.edge_index_dict:
                raise ValueError(f"Missing edge type in graph batch: {edge_type}")
            edge_index = self.edge_index_dict[edge_type]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"{edge_type} edge_index must be [2,E], got {tuple(edge_index.shape)}")
            edge_samples = self.edge_sample_indices[edge_type]
            if edge_samples.shape != (edge_index.shape[1],):
                raise ValueError(f"{edge_type} edge sample metadata shape mismatch")
            src_type, _, dst_type = edge_type
            if edge_index.numel():
                src_count = self.x_dict[src_type].shape[0]
                dst_count = self.x_dict[dst_type].shape[0]
                if (edge_index[0] < 0).any() or (edge_index[0] >= src_count).any():
                    raise ValueError(f"{edge_type} source edge index out of range")
                if (edge_index[1] < 0).any() or (edge_index[1] >= dst_count).any():
                    raise ValueError(f"{edge_type} destination edge index out of range")
                src_samples = self.node_sample_indices[src_type][edge_index[0]]
                dst_samples = self.node_sample_indices[dst_type][edge_index[1]]
                if not torch.equal(src_samples, dst_samples):
                    raise ValueError(f"{edge_type} contains cross-sample edges")
                if not torch.equal(src_samples, edge_samples):
                    raise ValueError(f"{edge_type} edge sample metadata mismatch")
            edge_attr = self.edge_attr_dict.get(edge_type)
            if edge_attr is None or edge_attr.shape != (edge_index.shape[1], 2):
                raise ValueError(f"{edge_type} edge_attr must be [E,2]")
            if not torch.isfinite(edge_attr).all():
                raise FloatingPointError(f"{edge_type} edge_attr contains non-finite values")

        hypothesis_nodes = self.x_dict[HYPOTHESIS]
        if hypothesis_nodes.shape[0] != self.batch_size * len(HYPOTHESIS_ORDER):
            raise ValueError("Hypothesis node count must be exactly B*4")


@dataclass
class HypothesisGraphOutput:
    """Phase-4 graph-reasoned hypothesis state output."""

    hypothesis_features: torch.Tensor
    hypothesis_mask: torch.Tensor
    graph_batch: GraphBatch
    graph_statistics: dict[str, Any]
    updated_node_features: dict[str, torch.Tensor] = field(default_factory=dict)
    intermediate_hypothesis_features: list[torch.Tensor] = field(default_factory=list)
    refined_visual_semantic: EvidenceSet | None = None
    refined_visual_forensic: EvidenceSet | None = None
    refined_face: EvidenceSet | None = None
    refined_text_alignment: EvidenceSet | None = None
    refined_text_context: EvidenceSet | None = None

    def validate(self) -> None:
        """Validate Phase-4 output without interpreting embeddings as logits."""

        if self.hypothesis_features.ndim != 3:
            raise ValueError("hypothesis_features must be [B,4,Dg]")
        batch, hypotheses, hidden_dim = self.hypothesis_features.shape
        if hypotheses != len(HYPOTHESIS_ORDER):
            raise ValueError("hypothesis_features must preserve [FS,FA,TS,TA] order")
        if self.hypothesis_mask.shape != (batch, hypotheses):
            raise ValueError("hypothesis_mask shape mismatch")
        if self.hypothesis_mask.dtype != torch.bool:
            raise ValueError("hypothesis_mask must be bool")
        if hidden_dim != self.graph_batch.hidden_dim:
            raise ValueError("hypothesis hidden_dim and graph hidden_dim mismatch")
        if not torch.isfinite(self.hypothesis_features).all():
            raise FloatingPointError("hypothesis_features contain non-finite values")
        if not bool(self.hypothesis_mask.all()):
            raise ValueError("All four hypothesis nodes must be valid")
        for state in self.intermediate_hypothesis_features:
            if state.shape != self.hypothesis_features.shape:
                raise ValueError("intermediate hypothesis state shape mismatch")
            if not torch.isfinite(state).all():
                raise FloatingPointError("intermediate hypothesis state contains non-finite values")
        for name in (
            "refined_visual_semantic",
            "refined_visual_forensic",
            "refined_face",
            "refined_text_alignment",
            "refined_text_context",
        ):
            evidence_set = getattr(self, name)
            if evidence_set is not None:
                evidence_set.validate()
                if evidence_set.features.shape[0] != batch:
                    raise ValueError(f"{name} batch size mismatch")
                if evidence_set.features.shape[-1] != hidden_dim:
                    raise ValueError(f"{name} hidden dimension mismatch")
        self.graph_batch.validate()
