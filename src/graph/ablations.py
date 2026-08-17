"""Production graph encoders used by controlled HERA ablations."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .hgt import HGTBackendUnavailable, _accumulate_edge_messages
from .schema import CANONICAL_EDGE_TYPES, EVIDENCE_NODE_TYPES, HYPOTHESIS, NODE_TYPES, EdgeType


class GATv2BackendUnavailable(RuntimeError):
    """Raised when the production PyG GATv2 backend cannot be imported."""


class GATv2GraphBlock(nn.Module):
    """A strong homogeneous GATv2 baseline with type and relation embeddings.

    Unlike HGT, one shared attention operator processes every node and edge.
    Type/relation embeddings keep the comparison strong without introducing
    relation-specific Q/K/V projections.
    """

    def __init__(self, *, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GATv2Conv
        except Exception as exc:  # pragma: no cover - only exercised without PyG
            raise GATv2BackendUnavailable(
                "torch_geometric is required for the production GATv2 ablation"
            ) from exc
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=int(heads),
            concat=False,
            dropout=float(dropout),
            edge_dim=hidden_dim,
            add_self_loops=True,
        )
        self.node_type_embeddings = nn.Parameter(torch.empty(len(NODE_TYPES), hidden_dim))
        self.relation_embeddings = nn.Parameter(torch.empty(len(CANONICAL_EDGE_TYPES), hidden_dim))
        self.edge_attr_encoders = nn.ModuleList(
            [nn.Linear(2, hidden_dim, bias=False) for _ in CANONICAL_EDGE_TYPES]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES})
        nn.init.normal_(self.node_type_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.relation_embeddings, mean=0.0, std=0.02)

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        flat, offsets = _flatten_nodes(x_dict)
        typed_parts = []
        for node_index, node_type in enumerate(NODE_TYPES):
            features = x_dict[node_type]
            typed_parts.append(
                features + self.node_type_embeddings[node_index].to(features.dtype).unsqueeze(0)
            )
        typed_flat = torch.cat(typed_parts, dim=0)

        global_edges: list[torch.Tensor] = []
        encoded_attrs: list[torch.Tensor] = []
        for relation_index, edge_type in enumerate(CANONICAL_EDGE_TYPES):
            indices = edge_index_dict[edge_type]
            if indices.shape[1] == 0:
                continue
            src_type, _, dst_type = edge_type
            global_edges.append(
                torch.stack(
                    [
                        indices[0] + offsets[src_type],
                        indices[1] + offsets[dst_type],
                    ],
                    dim=0,
                )
            )
            attrs = edge_attr_dict[edge_type].to(device=flat.device, dtype=flat.dtype)
            relation = self.relation_embeddings[relation_index].to(flat.dtype).unsqueeze(0)
            encoded_attrs.append(self.edge_attr_encoders[relation_index](attrs) + relation)

        if global_edges:
            edge_index = torch.cat(global_edges, dim=1)
            edge_attr = torch.cat(encoded_attrs, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=flat.device)
            edge_attr = flat.new_zeros((0, flat.shape[-1]))
        updated = self.conv(typed_flat, edge_index, edge_attr=edge_attr)
        result: dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            start = offsets[node_type]
            end = start + x_dict[node_type].shape[0]
            result[node_type] = self.norms[node_type](
                x_dict[node_type] + self.dropout(updated[start:end])
            )
        return result


class GATv2GraphEncoder(nn.Module):
    """Stack of production GATv2 blocks with the HGT-compatible interface."""

    def __init__(self, *, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GATv2GraphBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
                for _ in range(int(layers))
            ]
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        current = dict(x_dict)
        for layer in self.layers:
            current = layer(current, edge_index_dict, edge_attr_dict)
        return current


_EVIDENCE = "evidence"
_RELATES = (_EVIDENCE, "relates", _EVIDENCE)
_SUPPORTS = (_EVIDENCE, "supports", HYPOTHESIS)
_QUERIES = (HYPOTHESIS, "queries", _EVIDENCE)
_COARSE_NODE_TYPES = (_EVIDENCE, HYPOTHESIS)
_COARSE_EDGE_TYPES = (_RELATES, _SUPPORTS, _QUERIES)


class _CoarseHGTBlock(nn.Module):
    """HGT block after erasing all evidence-family and fine relation labels."""

    def __init__(self, *, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import HGTConv
        except Exception as exc:  # pragma: no cover - only exercised without PyG
            raise HGTBackendUnavailable(
                "torch_geometric is required for the homogeneous-evidence HGT ablation"
            ) from exc
        self.conv = HGTConv(
            in_channels={node_type: hidden_dim for node_type in _COARSE_NODE_TYPES},
            out_channels=hidden_dim,
            metadata=(list(_COARSE_NODE_TYPES), list(_COARSE_EDGE_TYPES)),
            heads=int(heads),
        )
        self.edge_attr_encoders = nn.ModuleDict(
            {
                "relates": nn.Linear(2, hidden_dim, bias=False),
                "supports": nn.Linear(2, hidden_dim, bias=False),
                "queries": nn.Linear(2, hidden_dim, bias=False),
            }
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in _COARSE_NODE_TYPES}
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        updated = self.conv(dict(x_dict), dict(edge_index_dict))
        edge_updates = {
            node_type: features.new_zeros(features.shape) for node_type, features in x_dict.items()
        }
        edge_counts = {
            node_type: features.new_zeros((features.shape[0], 1))
            for node_type, features in x_dict.items()
        }
        for edge_type in _COARSE_EDGE_TYPES:
            attrs = edge_attr_dict[edge_type]
            if attrs.numel() == 0 or not bool((attrs != 0).any()):
                continue
            dst_type = edge_type[2]
            messages = self.edge_attr_encoders[edge_type[1]](attrs.to(x_dict[dst_type].dtype))
            _accumulate_edge_messages(
                edge_updates[dst_type],
                edge_counts[dst_type],
                edge_index_dict[edge_type][1],
                messages,
            )
        result: dict[str, torch.Tensor] = {}
        for node_type in _COARSE_NODE_TYPES:
            update = updated.get(node_type)
            edge_update = edge_updates[node_type] / edge_counts[node_type].clamp_min(1.0)
            residual = x_dict[node_type] + self.dropout(edge_update)
            if update is not None:
                residual = residual + self.dropout(update)
            result[node_type] = self.norms[node_type](residual)
        return result


class HomogeneousEvidenceHGTEncoder(nn.Module):
    """Collapse five evidence families into one type before HGT reasoning."""

    def __init__(self, *, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _CoarseHGTBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
                for _ in range(int(layers))
            ]
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        edge_attr_dict: Mapping[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        coarse_x, coarse_edges, coarse_attrs, layout = _collapse_evidence_graph(
            x_dict, edge_index_dict, edge_attr_dict
        )
        current = coarse_x
        for layer in self.layers:
            current = layer(current, coarse_edges, coarse_attrs)
        return _expand_evidence_features(current, layout)


def _flatten_nodes(
    x_dict: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, int]]:
    offsets: dict[str, int] = {}
    cursor = 0
    parts: list[torch.Tensor] = []
    for node_type in NODE_TYPES:
        offsets[node_type] = cursor
        features = x_dict[node_type]
        parts.append(features)
        cursor += features.shape[0]
    return torch.cat(parts, dim=0), offsets


def _collapse_evidence_graph(
    x_dict: Mapping[str, torch.Tensor],
    edge_index_dict: Mapping[EdgeType, torch.Tensor],
    edge_attr_dict: Mapping[EdgeType, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    dict[EdgeType, torch.Tensor],
    dict[EdgeType, torch.Tensor],
    dict[str, tuple[int, int]],
]:
    evidence_parts: list[torch.Tensor] = []
    layout: dict[str, tuple[int, int]] = {}
    cursor = 0
    for node_type in EVIDENCE_NODE_TYPES:
        features = x_dict[node_type]
        layout[node_type] = (cursor, cursor + features.shape[0])
        cursor += features.shape[0]
        evidence_parts.append(features)
    coarse_x = {
        _EVIDENCE: torch.cat(evidence_parts, dim=0),
        HYPOTHESIS: x_dict[HYPOTHESIS],
    }
    grouped_indices: dict[EdgeType, list[torch.Tensor]] = {
        edge_type: [] for edge_type in _COARSE_EDGE_TYPES
    }
    grouped_attrs: dict[EdgeType, list[torch.Tensor]] = {
        edge_type: [] for edge_type in _COARSE_EDGE_TYPES
    }
    for edge_type in CANONICAL_EDGE_TYPES:
        indices = edge_index_dict[edge_type]
        src_type, _, dst_type = edge_type
        if src_type == HYPOTHESIS:
            coarse_type = _QUERIES
        elif dst_type == HYPOTHESIS:
            coarse_type = _SUPPORTS
        else:
            coarse_type = _RELATES
        if indices.shape[1] == 0:
            continue
        src_offset = 0 if src_type == HYPOTHESIS else layout[src_type][0]
        dst_offset = 0 if dst_type == HYPOTHESIS else layout[dst_type][0]
        grouped_indices[coarse_type].append(
            torch.stack([indices[0] + src_offset, indices[1] + dst_offset], dim=0)
        )
        grouped_attrs[coarse_type].append(edge_attr_dict[edge_type])

    coarse_edges: dict[EdgeType, torch.Tensor] = {}
    coarse_attrs: dict[EdgeType, torch.Tensor] = {}
    anchor = coarse_x[_EVIDENCE]
    for edge_type in _COARSE_EDGE_TYPES:
        coarse_edges[edge_type] = (
            torch.cat(grouped_indices[edge_type], dim=1)
            if grouped_indices[edge_type]
            else torch.empty((2, 0), dtype=torch.long, device=anchor.device)
        )
        coarse_attrs[edge_type] = (
            torch.cat(grouped_attrs[edge_type], dim=0)
            if grouped_attrs[edge_type]
            else anchor.new_zeros((0, 2))
        )
    return coarse_x, coarse_edges, coarse_attrs, layout


def _expand_evidence_features(
    coarse_x: Mapping[str, torch.Tensor],
    layout: Mapping[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    result = {HYPOTHESIS: coarse_x[HYPOTHESIS]}
    for node_type in EVIDENCE_NODE_TYPES:
        start, end = layout[node_type]
        result[node_type] = coarse_x[_EVIDENCE][start:end]
    return result
