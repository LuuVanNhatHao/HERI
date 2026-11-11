"""Reliability-aware heterogeneous graph transformer for canonical HERA."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn

from .schema import CANONICAL_EDGE_TYPES, NODE_TYPES


class HGTBackendUnavailable(RuntimeError):
    """Raised when the segment-softmax backend required by HERA is unavailable."""


def native_segment_softmax(
    logits: torch.Tensor, index: torch.Tensor, *, num_nodes: int
) -> torch.Tensor:
    """Stable per-destination/head softmax, mathematically identical to PyG.

    A native implementation also keeps small CPU verification independent of
    optional graph packages. Reductions and exponentials always use FP32.
    """
    values = logits.float()
    if values.numel() == 0:
        return values
    expanded = index[:, None].expand_as(values)
    maximum = values.new_full((num_nodes, values.shape[1]), -torch.inf)
    maximum.scatter_reduce_(
        0, expanded, values.detach(), reduce="amax", include_self=True
    )
    exponent = torch.exp(values - maximum[index])
    denominator = values.new_zeros((num_nodes, values.shape[1]))
    denominator.scatter_add_(0, expanded, exponent)
    return exponent / denominator[index].clamp_min(1e-8)


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

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        dropout: float,
        edge_types: Sequence[tuple[str, str, str]] = CANONICAL_EDGE_TYPES,
        node_types: Sequence[str] = NODE_TYPES,
        variant: str = "reliability_conditioned",
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(heads) != 0:
            raise ValueError("graph.hidden_dim must be divisible by graph.heads")
        self._segment_softmax = native_segment_softmax
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.edge_types = tuple(edge_types)
        self.node_types = tuple(node_types)
        self.variant = variant
        if variant not in {
            "standard",
            "edge_residual",
            "similarity_conditioned",
            "reliability_conditioned",
        }:
            raise ValueError(f"Unknown HGT attention variant {variant!r}")
        self.query = nn.ModuleDict(
            {
                node: nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for node in self.node_types
            }
        )
        self.key = nn.ModuleDict(
            {
                node: nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for node in self.node_types
            }
        )
        self.value = nn.ModuleDict(
            {
                node: nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for node in self.node_types
            }
        )
        self.output = nn.ModuleDict(
            {
                node: nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for node in self.node_types
            }
        )
        self.relation_key = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for _ in self.edge_types
            ]
        )
        self.relation_value = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                for _ in self.edge_types
            ]
        )
        self.edge_residual = (
            nn.ModuleList(
                [nn.Linear(2, self.hidden_dim, bias=False) for _ in self.edge_types]
            )
            if variant == "edge_residual"
            else None
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norms = nn.ModuleDict(
            {node: nn.LayerNorm(self.hidden_dim) for node in self.node_types}
        )

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
        # Project each node once per type; relation transforms remain distinct.
        queries_by_type = {
            node: self.query[node](features) for node, features in x_dict.items()
        }
        keys_by_type = {
            node: self.key[node](features) for node, features in x_dict.items()
        }
        values_by_type = {
            node: self.value[node](features) for node, features in x_dict.items()
        }
        for relation_index, edge_type in enumerate(self.edge_types):
            src_type, _, dst_type = edge_type
            indices = edge_index_dict[edge_type]
            if indices.numel() == 0:
                continue
            attrs = edge_attr_dict[edge_type].to(
                device=x_dict[src_type].device,
                dtype=torch.float32,
            )
            if attrs.shape != (indices.shape[1], 2):
                raise ValueError(f"{edge_type} edge attributes must be [E,2]")
            sources = indices[0]
            destinations = indices[1]
            queries = queries_by_type[dst_type][destinations].reshape(
                -1, self.heads, self.head_dim
            )
            keys = self.relation_key[relation_index](
                keys_by_type[src_type][sources]
            ).reshape(-1, self.heads, self.head_dim)
            values = self.relation_value[relation_index](
                values_by_type[src_type][sources]
            ).reshape(-1, self.heads, self.head_dim)
            if self.variant == "reliability_conditioned":
                confidence = attrs[:, 1].clamp(1e-8, 1.0)
            elif self.variant == "similarity_conditioned":
                confidence = ((1.0 + attrs[:, 0]) * 0.5).clamp(1e-8, 1.0)
            else:
                confidence = torch.ones_like(attrs[:, 1])
            attention_logits = torch.sum(
                queries.float() * keys.float(), dim=-1
            ) / math.sqrt(float(self.head_dim))
            if self.variant in {"similarity_conditioned", "reliability_conditioned"}:
                attention_logits = attention_logits + torch.log(
                    confidence + 1e-8
                ).unsqueeze(-1)
            attention = self._segment_softmax(
                attention_logits.float(),
                destinations,
                num_nodes=x_dict[dst_type].shape[0],
            )
            messages = (
                values.float()
                * attention.unsqueeze(-1)
                * confidence.unsqueeze(-1).unsqueeze(-1)
            )
            flattened = messages.reshape(-1, self.hidden_dim)
            if self.edge_residual is not None:
                flattened = (
                    flattened
                    + self.edge_residual[relation_index](attrs.to(values.dtype)).float()
                )
            relation_update = aggregated[dst_type].new_zeros(aggregated[dst_type].shape)
            relation_update.index_add_(
                0, destinations, flattened.to(relation_update.dtype)
            )
            aggregated[dst_type] = aggregated[dst_type] + relation_update

        result: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            update = self.output[node_type](aggregated[node_type])
            result[node_type] = self.norms[node_type](
                x_dict[node_type] + self.dropout(update)
            )
        return result


class HGTGraphEncoder(nn.Module):
    """Stack of canonical reliability-aware heterogeneous attention blocks."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        edge_types: Sequence[tuple[str, str, str]] = CANONICAL_EDGE_TYPES,
        variant: str = "reliability_conditioned",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EdgeAwareHGTGraphBlock(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_types=edge_types,
                    variant=variant,
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
