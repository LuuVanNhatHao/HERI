"""Reliability-aware heterogeneous graph transformer for canonical HERA."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn

from .schema import CANONICAL_EDGE_TYPES, NODE_TYPES


class HGTBackendUnavailable(RuntimeError):
    """Raised when the segment-softmax backend required by HERA is unavailable."""


def _accumulate_edge_messages(
    destination_updates: torch.Tensor,
    destination_counts: torch.Tensor,
    destinations: torch.Tensor,
    messages: torch.Tensor,
) -> None:
    """Accumulate typed edge updates without dtype-dependent ``index_add_`` failures.

    The helper remains shared with controlled legacy encoders. Canonical HGT does
    not use this residual path; confidence is incorporated directly in attention.
    """

    if destinations.numel() == 0:
        return
    destinations = destinations.to(device=destination_updates.device, dtype=torch.long)
    destination_updates.index_add_(
        0,
        destinations,
        messages.to(device=destination_updates.device, dtype=destination_updates.dtype),
    )
    ones = destination_counts.new_ones((destinations.shape[0], 1))
    destination_counts.index_add_(0, destinations, ones)


class EdgeAwareHGTGraphBlock(nn.Module):
    """Typed multi-head attention with relation projections and edge-confidence gates."""

    def __init__(self, *, hidden_dim: int, heads: int, dropout: float,
                 edge_types: Sequence[tuple[str, str, str]] = CANONICAL_EDGE_TYPES) -> None:
        super().__init__()
        if int(hidden_dim) % int(heads) != 0:
            raise ValueError("graph.hidden_dim must be divisible by graph.heads")
        try:
            from torch_geometric.utils import softmax as segment_softmax
        except Exception as exc:  # pragma: no cover - only exercised without PyG
            raise HGTBackendUnavailable(
                "torch_geometric is required for reliability-aware segment softmax"
            ) from exc
        self._segment_softmax = segment_softmax
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.edge_types = tuple(edge_types)
        self.query = nn.ModuleDict({node: nn.Linear(self.hidden_dim, self.hidden_dim) for node in NODE_TYPES})
        self.key = nn.ModuleDict({node: nn.Linear(self.hidden_dim, self.hidden_dim) for node in NODE_TYPES})
        self.value = nn.ModuleDict({node: nn.Linear(self.hidden_dim, self.hidden_dim) for node in NODE_TYPES})
        self.output = nn.ModuleDict({node: nn.Linear(self.hidden_dim, self.hidden_dim) for node in NODE_TYPES})
        self.relation_key = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in self.edge_types]
        )
        self.relation_value = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in self.edge_types]
        )
        self.edge_attention_bias = nn.ModuleList(
            [nn.Linear(2, self.heads, bias=False) for _ in self.edge_types]
        )
        self.relation_priority = nn.Parameter(torch.ones(len(self.edge_types), self.heads))
        self.dropout = nn.Dropout(float(dropout))
        self.norms = nn.ModuleDict({node: nn.LayerNorm(self.hidden_dim) for node in NODE_TYPES})
        for encoder in self.edge_attention_bias:
            nn.init.zeros_(encoder.weight)

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Mapping[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply confidence inside attention logits and directly gate relation messages."""

        aggregated = {
            node: features.new_zeros(features.shape)
            for node, features in x_dict.items()
        }
        for relation_index, edge_type in enumerate(self.edge_types):
            src_type, _, dst_type = edge_type
            indices = edge_index_dict[edge_type]
            if indices.numel() == 0:
                continue
            attrs = edge_attr_dict[edge_type].to(
                device=x_dict[src_type].device,
                dtype=x_dict[src_type].dtype,
            )
            if attrs.shape != (indices.shape[1], 2):
                raise ValueError(f"{edge_type} edge attributes must be [E,2]")
            sources = indices[0]
            destinations = indices[1]
            source_features = x_dict[src_type][sources]
            destination_features = x_dict[dst_type][destinations]
            queries = self.query[dst_type](destination_features).reshape(-1, self.heads, self.head_dim)
            keys = self.relation_key[relation_index](self.key[src_type](source_features)).reshape(
                -1, self.heads, self.head_dim
            )
            values = self.relation_value[relation_index](self.value[src_type](source_features)).reshape(
                -1, self.heads, self.head_dim
            )
            confidence = attrs[:, 1].float().clamp(0.0, 1.0).to(attrs.dtype)
            learned_bias = self.edge_attention_bias[relation_index](attrs)
            attention_logits = (
                torch.sum(queries * keys, dim=-1) / math.sqrt(float(self.head_dim))
                * self.relation_priority[relation_index].to(queries.dtype)
                + learned_bias
                + torch.log(confidence.clamp_min(1e-6)).unsqueeze(-1)
            )
            attention = self._segment_softmax(
                attention_logits.float(),
                destinations,
                num_nodes=x_dict[dst_type].shape[0],
            ).to(values.dtype)
            messages = values * attention.unsqueeze(-1) * confidence.unsqueeze(-1).unsqueeze(-1)
            flattened = messages.reshape(-1, self.hidden_dim)
            relation_update = aggregated[dst_type].new_zeros(aggregated[dst_type].shape)
            relation_update.index_add_(0, destinations, flattened.to(relation_update.dtype))
            aggregated[dst_type] = aggregated[dst_type] + relation_update

        result: dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            update = self.output[node_type](aggregated[node_type])
            result[node_type] = self.norms[node_type](x_dict[node_type] + self.dropout(update))
        return result


class HGTGraphEncoder(nn.Module):
    """Stack of canonical reliability-aware heterogeneous attention blocks."""

    def __init__(self, *, hidden_dim: int, layers: int, heads: int, dropout: float,
                 edge_types: Sequence[tuple[str, str, str]] = CANONICAL_EDGE_TYPES) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EdgeAwareHGTGraphBlock(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_types=edge_types,
                )
                for _ in range(int(layers))
            ]
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Mapping[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        current = dict(x_dict)
        for layer in self.layers:
            current = layer(current, edge_index_dict, edge_attr_dict)
        return current


# Backward-compatible name for imports while making the implementation explicit.
HGTGraphBlock = EdgeAwareHGTGraphBlock
