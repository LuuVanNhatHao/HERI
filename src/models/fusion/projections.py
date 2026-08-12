"""Projection utilities that adapt native backbone dimensions lazily."""

from __future__ import annotations

import torch
from torch import nn


class LazyFeatureProjection(nn.Module):
    """Lazy Linear -> LayerNorm -> activation -> dropout projection."""

    def __init__(self, *, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features of shape [..., C] to [..., output_dim]."""

        return self.net(features)


class MLPBlock(nn.Module):
    """Small configurable MLP block for baseline heads/fusion."""

    def __init__(self, *, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
