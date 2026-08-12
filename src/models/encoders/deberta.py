"""DeBERTa contextual text encoder wrapper."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.compat.transformers_compat import load_auto_model, load_fast_tokenizer

from .common import freeze_module, masked_mean, unfreeze_last_blocks
from ..outputs import DeBERTaOutput


class DeBERTaEncoder(nn.Module):
    """Production DeBERTa wrapper using Phase-1 fast-tokenizer alignment."""

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
    def from_config(cls, config: Mapping[str, Any]) -> "DeBERTaEncoder":
        """Load the configured DeBERTa model and fast tokenizer."""

        name = str(config["name"])
        runtime = config.get("runtime", {})
        model = load_auto_model(
            name=name,
            stage="DeBERTa",
            revision=config.get("revision"),
            local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            torch_dtype=runtime.get("model_dtype"),
            attn_implementation=runtime.get("attn_implementation"),
        )
        tokenizer = load_fast_tokenizer(
            name=name,
            use_fast=True,
            local_files_only=bool(config.get("local_files_only", False)),
            extra_kwargs={"revision": config.get("revision")} if config.get("revision") else None,
        )
        return cls(model=model, tokenizer=tokenizer, model_name=name)

    def freeze_all(self) -> None:
        """Freeze all DeBERTa parameters."""

        freeze_module(self.model)

    def unfreeze_last_blocks(self, count: int) -> int:
        """Unfreeze final DeBERTa blocks via compatibility introspection."""

        return unfreeze_last_blocks(self.model, count, stage_name="DeBERTa")

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
    raise ValueError("DeBERTa output missing last_hidden_state")
