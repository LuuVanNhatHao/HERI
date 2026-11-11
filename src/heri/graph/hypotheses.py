"""Learnable FS/FA/TS/TA hypothesis node initialization."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .schema import HypothesisSpec


class HypothesisNodeInitializer(nn.Module):
    """Own the ontology-instantiated global learnable hypothesis embeddings."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        init_std = float(config.get("hypothesis_init_std", 0.02))
        self.spec = HypothesisSpec.from_mapping(config.get("hypotheses"))
        self.embeddings = nn.Parameter(torch.empty(len(self.spec.order), hidden_dim))
        nn.init.normal_(self.embeddings, mean=0.0, std=init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return ``[B,K,Dg]`` hypothesis states in configured ontology order."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
