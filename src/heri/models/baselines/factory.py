"""Factory for controlled baseline models."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from heri.models.encoders.deberta import ContextTextEncoder
from heri.models.encoders.dinov3 import DINOv3Encoder
from heri.models.encoders.siglip2 import SigLIP2Encoder

from .b0_global import B0GlobalFusionBaseline
from .b1_cross_attention import B1CrossAttentionBaseline


def build_backbones(config: Mapping[str, Any]) -> dict[str, nn.Module]:
    """Load all three configured pretrained backbones."""

    backbones = config["backbones"]
    runtime = config.get("runtime", {})
    merged = {}
    for name, backbone_config in backbones.items():
        cfg = dict(backbone_config)
        backbone_runtime = dict(runtime)
        if cfg.get("attn_implementation") not in (None, ""):
            backbone_runtime["attn_implementation"] = cfg["attn_implementation"]
        cfg["runtime"] = backbone_runtime
        merged[name] = cfg
    return {
        "siglip2": SigLIP2Encoder.from_config(merged["siglip2"]),
        "dinov3": DINOv3Encoder.from_config(merged["dinov3"]),
        "deberta": ContextTextEncoder.from_config(merged["deberta"]),
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
        "hypothesis_order": tuple(
            model_config.get("hypotheses", {}).get("order", ("FS", "FA", "TS", "TA"))
        ),
        "image_hypothesis_names": tuple(
            model_config.get("hypotheses", {}).get("image", ("FS", "FA"))
        ),
    }
    if baseline_type == "b0_global":
        return B0GlobalFusionBaseline(
            fusion_hidden_dim=int(model_config["fusion_hidden_dim"]),
            **common,
        )
    if baseline_type == "b1_cross_attention":
        return B1CrossAttentionBaseline(
            binary_aggregation=str(model_config.get("binary_aggregation", "independent_head")),
            num_layers=int(model_config["num_attention_layers"]),
            num_heads=int(model_config["num_heads"]),
            **common,
        )
    raise ValueError(f"Unknown baseline_type: {baseline_type!r}")
