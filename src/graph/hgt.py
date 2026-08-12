"""Production PyG HGT wrapper for Phase-4 graph reasoning."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .schema import NODE_TYPES, graph_metadata


class HGTBackendUnavailable(RuntimeError):
    """Raised when the configured production HGT backend is unavailable."""


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

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run one typed HGT update."""

        updated = self.conv(dict(x_dict), dict(edge_index_dict))
        result: dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            update = updated.get(node_type)
            if update is None:
                result[node_type] = x_dict[node_type]
            else:
                result[node_type] = self.norms[node_type](x_dict[node_type] + self.dropout(update))
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
    ) -> dict[str, torch.Tensor]:
        """Run all configured HGT layers."""

        current = dict(x_dict)
        for layer in self.layers:
            current = layer(current, edge_index_dict)
        return current
