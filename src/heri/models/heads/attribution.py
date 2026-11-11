"""FS/FA/TS/TA multilabel attribution heads."""

from __future__ import annotations

import torch
from torch import nn

from heri.graph.schema import HYPOTHESIS_ORDER


class AttributionHead(nn.Module):
    """Map fused representation to four independent atomic logits."""

    def __init__(self, *, input_dim: int, num_labels: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.linear(fused)


class HypothesisAttributionHead(nn.Module):
    """Map each graph hypothesis state to its own atomic manipulation logit."""

    def __init__(self, *, hidden_dim: int, dropout: float = 0.0,
                 hypothesis_order: tuple[str, ...] = HYPOTHESIS_ORDER) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.order = tuple(hypothesis_order)
        if not self.order or len(set(self.order)) != len(self.order):
            raise ValueError("hypothesis_order must be non-empty and unique")
        self.readouts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.hidden_dim, 1),
                )
                for _ in self.order
            ]
        )

    def forward(
        self,
        hypothesis_features: torch.Tensor,
        hypothesis_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one attribution logit per configured hypothesis."""

        if hypothesis_features.ndim != 3:
            raise ValueError(f"hypothesis_features must be [B,K,D], got {tuple(hypothesis_features.shape)}")
        batch, hypotheses, hidden_dim = hypothesis_features.shape
        if hypotheses != len(self.order):
            raise ValueError("hypothesis_features must preserve configured hypothesis order")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}")
        if hypothesis_mask is not None:
            if hypothesis_mask.shape != (batch, hypotheses):
                raise ValueError("hypothesis_mask shape mismatch")
            if hypothesis_mask.dtype != torch.bool:
                raise ValueError("hypothesis_mask must be bool")
            if not bool(hypothesis_mask.all()):
                raise ValueError("All HERA hypothesis states must be valid")
        logits = [
            readout(hypothesis_features[:, hypothesis_index, :]).squeeze(-1)
            for hypothesis_index, readout in enumerate(self.readouts)
        ]
        return torch.stack(logits, dim=1)
