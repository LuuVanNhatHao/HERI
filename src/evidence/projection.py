"""Family-specific evidence projection modules."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class EvidenceProjection(nn.Module):
    """Lightweight Linear -> activation -> LayerNorm -> Dropout adapter."""

    def __init__(
        self,
        *,
        output_dim: int,
        input_dim: int | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        linear: nn.Module = nn.Linear(input_dim, output_dim) if input_dim is not None else nn.LazyLinear(output_dim)
        modules: list[nn.Module] = [linear, _activation(activation)]
        if layer_norm:
            modules.append(nn.LayerNorm(output_dim))
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)
        self.output_dim = int(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features of shape [B,N,C] into the common evidence space."""

        if features.ndim != 3:
            raise ValueError(f"EvidenceProjection input must be [B,N,C], got {tuple(features.shape)}")
        return self.net(features)


class EvidenceProjectionBank(nn.Module):
    """Separate projectors for canonical Phase-3 evidence families."""

    families = ("visual_semantic", "visual_forensic", "face", "text_alignment", "text_context")

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        dropout = float(config.get("dropout", 0.0))
        activation = str(config.get("activation", "gelu"))
        layer_norm = bool(config.get("layer_norm", True))
        input_dims = config.get("input_dims", {}) or {}
        self.projectors = nn.ModuleDict(
            {
                family: EvidenceProjection(
                    input_dim=_optional_int(input_dims.get(family)),
                    output_dim=hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
                for family in self.families
            }
        )

    def __getitem__(self, family: str) -> EvidenceProjection:
        return self.projectors[family]


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported evidence projection activation: {name!r}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
