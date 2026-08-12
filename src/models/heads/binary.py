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
