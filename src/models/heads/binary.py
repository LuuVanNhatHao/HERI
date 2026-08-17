"""Direct binary Real/Fake head allowed for Phase-2 baselines."""

from __future__ import annotations

import torch
from torch import nn


class BinaryHead(nn.Module):
    """Map fused representation to one fake logit."""

    def __init__(self, *, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.linear(fused).squeeze(-1)


class IndependentHypothesisBinaryHead(nn.Module):
    """Predict authenticity from graph states without using atomic logits."""

    def __init__(self, *, hidden_dim: int, hypotheses: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        input_dim = int(hidden_dim) * int(hypotheses)
        self.hidden_dim = int(hidden_dim)
        self.hypotheses = int(hypotheses)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, hypothesis_features: torch.Tensor) -> torch.Tensor:
        if hypothesis_features.ndim != 3:
            raise ValueError("hypothesis_features must be [B,4,D]")
        if hypothesis_features.shape[1:] != (self.hypotheses, self.hidden_dim):
            raise ValueError(
                "Independent binary head expected hypothesis features "
                f"[B,{self.hypotheses},{self.hidden_dim}]"
            )
        return self.network(hypothesis_features.flatten(start_dim=1)).squeeze(-1)
