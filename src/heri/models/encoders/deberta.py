"""Generic contextual text encoder wrapper (legacy module path)."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from heri.compat.transformers_compat import load_auto_model, load_fast_tokenizer

from .common import freeze_module, masked_mean, unfreeze_last_blocks
from ..outputs import DeBERTaOutput


class ContextTextEncoder(nn.Module):
    """Production AutoModel wrapper with fast-tokenizer alignment."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        model_name: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ContextTextEncoder":
        """Load the configured encoder-only text model and fast tokenizer."""

        name = str(config["name"])
        runtime = config.get("runtime", {})
        model = load_auto_model(
            name=name,
            stage="ContextTextEncoder",
            revision=config.get("revision"),
            local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            torch_dtype=runtime.get("model_dtype"),
            attn_implementation=runtime.get("attn_implementation"),
        )
        configure_reference_compile(
            model,
            requested=runtime.get("reference_compile"),
        )
        tokenizer = load_fast_tokenizer(
            name=name,
            use_fast=True,
            local_files_only=bool(config.get("local_files_only", False)),
            extra_kwargs={"revision": config.get("revision")} if config.get("revision") else None,
        )
        return cls(model=model, tokenizer=tokenizer, model_name=name)

    def freeze_all(self) -> None:
        """Freeze all contextual text encoder parameters."""

        freeze_module(self.model)

    def unfreeze_last_blocks(self, count: int) -> int:
        """Unfreeze final text encoder blocks via compatibility introspection."""

        return unfreeze_last_blocks(self.model, count, stage_name="ContextTextEncoder")

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> DeBERTaOutput:
        """Encode tokenizer-aligned DeBERTa subwords."""

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        except TypeError:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_features = _extract_last_hidden_state(outputs)
        mask = attention_mask.to(torch.bool)
        return DeBERTaOutput(
            token_features=token_features,
            attention_mask=mask,
            global_feature=masked_mean(token_features, mask),
        )


def _extract_last_hidden_state(outputs: Any) -> torch.Tensor:
    value = getattr(outputs, "last_hidden_state", None)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(outputs, Mapping):
        value = outputs.get("last_hidden_state")
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError("Context text encoder output missing last_hidden_state")


# Backward-compatible import for DGM4 and existing checkpoints/configuration code.
DeBERTaEncoder = ContextTextEncoder


def configure_reference_compile(
    model: nn.Module,
    *,
    requested: Any,
) -> bool | None:
    """Set optional encoder-internal reference compilation before first forward.

    ModernBERT may automatically compile its embedding and MLP helpers when
    Triton is available. A graph first compiled while Stage A is frozen can
    retain that frozen parameter contract after Stage B unfreezes the model,
    leaving the token embedding without gradients. Explicitly disabling that
    optional compile path keeps staged full fine-tuning correct; gradient
    checkpointing and the configured attention implementation remain active.
    """

    if requested is None:
        return None
    model_config = getattr(model, "config", None)
    if model_config is None or not hasattr(model_config, "reference_compile"):
        raise RuntimeError(
            "runtime.reference_compile was configured, but the contextual "
            "text backbone does not expose config.reference_compile"
        )
    enabled = bool(requested)
    model_config.reference_compile = enabled
    return enabled
