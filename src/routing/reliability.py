"""Family-local reliability estimators for Phase-6 routing."""

from __future__ import annotations

import torch
from torch import nn


class EvidenceReliabilityEstimator(nn.Module):
    """LayerNorm -> small MLP -> scalar reliability logit per evidence node."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        initial_bias: float,
        output_weight_std: float,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if float(dropout) < 0:
            raise ValueError("dropout must be non-negative")
        if float(output_weight_std) < 0:
            raise ValueError("output_weight_std must be non-negative")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.input_dim)
        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters(initial_bias=float(initial_bias), output_weight_std=float(output_weight_std))

    def reset_parameters(self, *, initial_bias: float, output_weight_std: float) -> None:
        """Initialize toward high-neutral evidence preservation from YAML bias."""

        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=output_weight_std)
        nn.init.constant_(self.output.bias, initial_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return reliability logits [B,N] for features [B,N,D]."""

        if features.ndim != 3:
            raise ValueError(f"features must be [B,N,D], got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {features.shape[-1]}")
        if not torch.isfinite(features).all():
            raise FloatingPointError("Reliability estimator received non-finite features")
        hidden = self.activation(self.hidden(self.norm(features)))
        return self.output(self.dropout(hidden)).squeeze(-1)
