"""Graph validation and statistics helpers."""

from __future__ import annotations

from typing import Any

import torch

from .outputs import GraphBatch, HypothesisGraphOutput
from .schema import EVIDENCE_NODE_TYPES, HYPOTHESIS, HYPOTHESIS_ORDER


def validate_graph_batch(graph: GraphBatch) -> None:
    """Fail fast on graph contract violations."""

    graph.validate()


def validate_hypothesis_output(output: HypothesisGraphOutput) -> None:
    """Fail fast on Phase-4 output contract violations."""

    output.validate()


def graph_statistics(graph: GraphBatch) -> dict[str, Any]:
    """Return concise graph diagnostics without reporting performance metrics."""

    node_counts = {
        node_type: int(features.shape[0])
        for node_type, features in graph.x_dict.items()
    }
    node_counts_per_sample = {
        node_type: [
            int((graph.node_sample_indices[node_type] == sample_index).sum().item())
            for sample_index in range(graph.batch_size)
        ]
        for node_type in graph.x_dict
    }
    edge_counts = {
        _edge_name(edge_type): int(edge_index.shape[1])
        for edge_type, edge_index in graph.edge_index_dict.items()
    }
    edge_counts_per_sample = {
        _edge_name(edge_type): [
            int((graph.edge_sample_indices[edge_type] == sample_index).sum().item())
            for sample_index in range(graph.batch_size)
        ]
        for edge_type in graph.edge_index_dict
    }
    return {
        "batch_size": graph.batch_size,
        "hidden_dim": graph.hidden_dim,
        "node_counts": node_counts,
        "node_counts_per_sample": node_counts_per_sample,
        "edge_counts": edge_counts,
        "edge_counts_per_sample": edge_counts_per_sample,
        "mean_degree": _mean_degree(graph),
        "isolated_evidence_nodes": _isolated_evidence_nodes(graph),
        "valid_hypothesis_nodes": int(graph.x_dict[HYPOTHESIS].shape[0]),
        "hypothesis_order": list(
            graph.metadata.get("hypothesis_order", HYPOTHESIS_ORDER)
        ),
    }


def _mean_degree(graph: GraphBatch) -> float:
    total_nodes = sum(features.shape[0] for features in graph.x_dict.values())
    total_edges = sum(edge_index.shape[1] for edge_index in graph.edge_index_dict.values())
    return float(total_edges / total_nodes) if total_nodes else 0.0


def _isolated_evidence_nodes(graph: GraphBatch) -> dict[str, int]:
    isolated: dict[str, int] = {}
    for node_type in EVIDENCE_NODE_TYPES:
        count = graph.x_dict[node_type].shape[0]
        incident = torch.zeros(count, dtype=torch.bool, device=graph.x_dict[node_type].device)
        for edge_type, edge_index in graph.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            if edge_index.numel() == 0:
                continue
            if src_type == node_type:
                incident[edge_index[0]] = True
            if dst_type == node_type:
                incident[edge_index[1]] = True
        isolated[node_type] = int((~incident).sum().item())
    return isolated


def _edge_name(edge_type: tuple[str, str, str]) -> str:
    return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
