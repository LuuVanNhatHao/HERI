"""Factory for Phase-2 controlled baseline models."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.encoders.deberta import DeBERTaEncoder
from src.models.encoders.dinov3 import DINOv3Encoder
from src.models.encoders.siglip2 import SigLIP2Encoder

from .b0_global import B0GlobalFusionBaseline
from .b1_cross_attention import B1CrossAttentionBaseline


def build_backbones(config: Mapping[str, Any]) -> dict[str, nn.Module]:
    """Load all three configured pretrained backbones."""

    backbones = config["backbones"]
    runtime = config.get("runtime", {})
    merged = {}
    for name, backbone_config in backbones.items():
        cfg = dict(backbone_config)
        cfg["runtime"] = runtime
        merged[name] = cfg
    return {
        "siglip2": SigLIP2Encoder.from_config(merged["siglip2"]),
        "dinov3": DINOv3Encoder.from_config(merged["dinov3"]),
        "deberta": DeBERTaEncoder.from_config(merged["deberta"]),
    }


def build_baseline_model(
    config: Mapping[str, Any],
    *,
    load_backbones: bool = True,
) -> nn.Module:
    """Build B0 or B1 from YAML-derived configuration."""

    model_config = config["model"]
    baseline_type = str(model_config["baseline_type"])
    backbones = build_backbones(config) if load_backbones else {}
    common = {
        "fusion_dim": int(model_config["fusion_dim"]),
        "head_hidden_dim": int(model_config["head_hidden_dim"]),
        "dropout": float(model_config["dropout"]),
        "siglip2": backbones.get("siglip2"),
        "dinov3": backbones.get("dinov3"),
        "deberta": backbones.get("deberta"),
    }
    if baseline_type == "b0_global":
        return B0GlobalFusionBaseline(
            fusion_hidden_dim=int(model_config["fusion_hidden_dim"]),
            **common,
        )
    if baseline_type == "b1_cross_attention":
        return B1CrossAttentionBaseline(
            num_layers=int(model_config["num_attention_layers"]),
            num_heads=int(model_config["num_heads"]),
            **common,
        )
    raise ValueError(f"Unknown baseline_type: {baseline_type!r}")
