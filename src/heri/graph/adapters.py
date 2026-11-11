"""Typed adapters for graph feature dimensions."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import torch
from torch import nn

from heri.evidence.outputs import HeterogeneousEvidenceBatch

from .schema import EVIDENCE_NODE_TYPES, NODE_TYPES


class TypedGraphFeatureAdapters(nn.Module):
    """Adapt per-node-type features to graph.hidden_dim when required."""

    def __init__(self, *, hidden_dim: int, input_dims: Mapping[str, int | None] | None = None) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        input_dims = input_dims or {}
        modules: dict[str, nn.Module] = {}
        for node_type in NODE_TYPES:
            input_dim = input_dims.get(node_type)
            if input_dim is None or int(input_dim) == self.hidden_dim:
                modules[node_type] = nn.Identity()
            else:
                modules[node_type] = nn.Sequential(
                    nn.Linear(int(input_dim), self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                )
        self.adapters = nn.ModuleDict(modules)

    def forward(self, x_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return adapted node features by type."""

        return {node_type: self.adapters[node_type](x_dict[node_type]) for node_type in NODE_TYPES}


class TypedEvidenceFeatureAdapters(nn.Module):
    """Adapt evidence sets to graph.hidden_dim before graph building."""

    def __init__(self, *, hidden_dim: int, input_dims: Mapping[str, int | None] | None = None) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        input_dims = input_dims or {}
        modules: dict[str, nn.Module] = {}
        for node_type in EVIDENCE_NODE_TYPES:
            input_dim = input_dims.get(node_type)
            if input_dim is None or int(input_dim) == self.hidden_dim:
                modules[node_type] = nn.Identity()
            else:
                modules[node_type] = nn.Sequential(
                    nn.Linear(int(input_dim), self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                )
        self.adapters = nn.ModuleDict(modules)

    def forward(self, evidence: HeterogeneousEvidenceBatch) -> HeterogeneousEvidenceBatch:
        """Return a shallow evidence copy with adapted graph-node features."""

        return HeterogeneousEvidenceBatch(
            visual_semantic=replace(
                evidence.visual_semantic,
                features=self.adapters["visual_semantic"](evidence.visual_semantic.features),
            ),
            visual_forensic=replace(
                evidence.visual_forensic,
                features=self.adapters["visual_forensic"](evidence.visual_forensic.features),
            ),
            text_alignment=replace(
                evidence.text_alignment,
                features=self.adapters["text_alignment"](evidence.text_alignment.features),
            ),
            text_context=replace(
                evidence.text_context,
                features=self.adapters["text_context"](evidence.text_context.features),
            ),
            cross_modal_metadata=evidence.cross_modal_metadata,
        )
