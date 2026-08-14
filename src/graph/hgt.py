"""Production PyG HGT wrapper for Phase-4 graph reasoning."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .schema import CANONICAL_EDGE_TYPES, NODE_TYPES, graph_metadata


class HGTBackendUnavailable(RuntimeError):
    """Raised when the configured production HGT backend is unavailable."""


def _accumulate_edge_messages(
    accumulator: torch.Tensor,
    counts: torch.Tensor,
    destinations: torch.Tensor,
    messages: torch.Tensor,
) -> None:
    """Accumulate autocast messages into the residual stream's stable dtype."""

    destinations = destinations.to(device=accumulator.device)
    messages = messages.to(device=accumulator.device, dtype=accumulator.dtype)
    accumulator.index_add_(0, destinations, messages)
    counts.index_add_(
        0,
        destinations,
        counts.new_ones((destinations.numel(), 1)),
    )


class HGTGraphBlock(nn.Module):
    """One HGTConv update followed by residual, dropout, and LayerNorm."""

    def __init__(self, *, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import HGTConv
        except Exception as exc:  # pragma: no cover - exercised only without optional backend
            raise HGTBackendUnavailable(
                "torch_geometric is required for Phase-4 production HGT. "
                "Install requirements.txt and rerun graph server validation."
            ) from exc
        self.conv = HGTConv(
            in_channels={node_type: hidden_dim for node_type in NODE_TYPES},
            out_channels=hidden_dim,
            metadata=graph_metadata(),
            heads=heads,
        )
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES})
        self.edge_attr_encoders = nn.ModuleList(
            [nn.Linear(2, hidden_dim, bias=False) for _ in CANONICAL_EDGE_TYPES]
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Mapping[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run one typed HGT update with explicit continuous edge evidence."""

        updated = self.conv(dict(x_dict), dict(edge_index_dict))
        edge_updates = {
            node_type: features.new_zeros(features.shape)
            for node_type, features in x_dict.items()
        }
        edge_counts = {
            node_type: features.new_zeros((features.shape[0], 1))
            for node_type, features in x_dict.items()
        }
        for edge_index, edge_type in enumerate(CANONICAL_EDGE_TYPES):
            attrs = edge_attr_dict[edge_type]
            indices = edge_index_dict[edge_type]
            if attrs.numel() == 0 or not bool((attrs != 0).any()):
                continue
            dst_type = edge_type[2]
            messages = self.edge_attr_encoders[edge_index](attrs.to(x_dict[dst_type].dtype))
            destinations = indices[1]
            _accumulate_edge_messages(
                edge_updates[dst_type],
                edge_counts[dst_type],
                destinations,
                messages,
            )
        result: dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            update = updated.get(node_type)
            edge_update = edge_updates[node_type] / edge_counts[node_type].clamp_min(1.0)
            if update is None:
                result[node_type] = self.norms[node_type](x_dict[node_type] + self.dropout(edge_update))
            else:
                result[node_type] = self.norms[node_type](
                    x_dict[node_type] + self.dropout(update) + self.dropout(edge_update)
                )
        return result


class HGTGraphEncoder(nn.Module):
    """A shallow stack of real HGT graph blocks."""

    def __init__(self, *, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HGTGraphBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
                for _ in range(int(layers))
            ]
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Mapping[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run all configured HGT layers."""

        current = dict(x_dict)
        for layer in self.layers:
            current = layer(current, edge_index_dict, edge_attr_dict)
        return current
