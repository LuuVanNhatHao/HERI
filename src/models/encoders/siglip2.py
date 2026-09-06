"""SigLIP2 semantic multimodal encoder wrapper."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.compat.transformers_compat import load_auto_model, load_auto_tokenizer

from .common import (
    freeze_module,
    infer_square_grid,
    masked_mean,
    normalize_image_tensor,
    pixel_mask_to_patch_mask,
    unfreeze_last_blocks,
)
from ..outputs import PatchGrid, SigLIP2Output


class SigLIP2Encoder(nn.Module):
    """Production SigLIP2 wrapper with separated geometry and normalization."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        max_length: int,
        model_name: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)
        self.max_length = int(max_length)
        self.model_name = model_name
        if self.max_length <= 0:
            raise ValueError("SigLIP2 text_max_length must be positive")
        text_capacity = _text_position_capacity(model)
        if text_capacity is not None and self.max_length > text_capacity:
            raise ValueError(
                f"Configured SigLIP2 text_max_length={self.max_length} exceeds "
                f"{model_name}'s max_position_embeddings={text_capacity}. "
                "Keep SigLIP2 text_max_length within the pretrained text tower's "
                "capacity; tokenizer.max_length may remain larger for the separate "
                "text-grounding encoder."
            )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SigLIP2Encoder":
        """Load the configured SigLIP2 model/tokenizer."""

        name = str(config["name"])
        revision = config.get("revision")
        runtime = config.get("runtime", {})
        model = load_auto_model(
            name=name,
            stage="SigLIP2",
            revision=revision,
            local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            torch_dtype=runtime.get("model_dtype"),
            attn_implementation=runtime.get("attn_implementation"),
        )
        tokenizer = load_auto_tokenizer(
            name=name,
            stage="SigLIP2 text",
            revision=revision,
            local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            use_fast=True,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_mean=config["image_mean"],
            image_std=config["image_std"],
            max_length=int(config["text_max_length"]),
            model_name=name,
        )

    def freeze_all(self) -> None:
        """Freeze all SigLIP2 parameters."""

        freeze_module(self.model)

    def unfreeze_last_blocks(self, count: int) -> int:
        """Unfreeze final SigLIP2 blocks via compatibility introspection."""

        return unfreeze_last_blocks(self.model, count, stage_name="SigLIP2")

    def tokenize_texts(self, texts: list[str], *, device: torch.device) -> dict[str, torch.Tensor]:
        """Tokenize text for SigLIP2 semantic text features."""

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in encoding.items()}

    def forward(
        self,
        *,
        image: torch.Tensor,
        texts: list[str],
        words: list[list[str]] | None = None,
        pixel_mask: torch.Tensor | None = None,
    ) -> SigLIP2Output:
        """Encode Phase-1 letterboxed images and raw texts."""

        pixel_values = normalize_image_tensor(
            image,
            mean=self.image_mean,
            std=self.image_std,
        )
        text_encoding = self.tokenizer(
            words if words is not None else texts,
            is_split_into_words=True if words is not None else False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        text_word_ids = _batch_word_ids(text_encoding, batch_size=len(texts), device=image.device)
        text_inputs = {
            key: value.to(image.device)
            for key, value in text_encoding.items()
            if isinstance(value, torch.Tensor)
        }
        try:
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                return_dict=True,
            )
        except TypeError:
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )

        raw_image_tokens = _extract_nested_tensor(outputs, ("vision_model_output", "last_hidden_state"))
        text_tokens = _extract_nested_tensor(outputs, ("text_model_output", "last_hidden_state"))
        image_global = _extract_optional_tensor(outputs, "image_embeds")
        text_global = _extract_optional_tensor(outputs, "text_embeds")
        image_tokens, image_grid, image_special_token_count = _validated_local_image_tokens(
            raw_image_tokens,
            image=image,
        )
        text_mask = text_inputs.get("attention_mask")
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        else:
            text_mask = text_mask.to(torch.bool)
        image_mask = pixel_mask_to_patch_mask(
            pixel_mask,
            grid_height=image_grid.height,
            grid_width=image_grid.width,
            batch_size=image_tokens.shape[0],
            device=image_tokens.device,
        )
        if image_global is None:
            image_global = masked_mean(image_tokens, image_mask)
        if text_global is None:
            text_global = masked_mean(text_tokens, text_mask)
        return SigLIP2Output(
            image_tokens=image_tokens,
            image_mask=image_mask,
            image_global=image_global,
            text_tokens=text_tokens,
            text_mask=text_mask,
            text_global=text_global,
            image_grid=image_grid,
            image_special_token_count=image_special_token_count,
            text_word_ids=text_word_ids,
        )


def _batch_word_ids(encoding: Any, *, batch_size: int, device: torch.device) -> torch.Tensor | None:
    """Extract fast-tokenizer word IDs without depending on supervision labels."""

    if not hasattr(encoding, "word_ids"):
        return None
    rows: list[list[int]] = []
    for batch_index in range(batch_size):
        try:
            values = encoding.word_ids(batch_index=batch_index)
        except TypeError:
            if batch_size != 1:
                return None
            values = encoding.word_ids()
        rows.append([-1 if value is None else int(value) for value in values])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _text_position_capacity(model: nn.Module) -> int | None:
    """Return the pretrained text tower's positional capacity when advertised."""

    config = getattr(model, "config", None)
    if config is None:
        return None
    text_config = (
        config.get("text_config")
        if isinstance(config, Mapping)
        else getattr(config, "text_config", None)
    )
    if text_config is None:
        return None
    capacity = (
        text_config.get("max_position_embeddings")
        if isinstance(text_config, Mapping)
        else getattr(text_config, "max_position_embeddings", None)
    )
    if capacity is None:
        return None
    parsed = int(capacity)
    return parsed if parsed > 0 else None


def _extract_optional_tensor(outputs: Any, name: str) -> torch.Tensor | None:
    value = getattr(outputs, name, None)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(outputs, Mapping):
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _extract_nested_tensor(outputs: Any, path: tuple[str, str]) -> torch.Tensor:
    current: Any = outputs
    for part in path:
        next_value = getattr(current, part, None)
        if next_value is None and isinstance(current, Mapping):
            next_value = current.get(part)
        current = next_value
        if current is None:
            raise ValueError(f"SigLIP2 output missing {'.'.join(path)}")
    if not isinstance(current, torch.Tensor):
        raise ValueError(f"SigLIP2 output {'.'.join(path)} is not a tensor")
    return current


def _validated_local_image_tokens(
    tokens: torch.Tensor,
    *,
    image: torch.Tensor,
) -> tuple[torch.Tensor, PatchGrid, int]:
    """Return spatial local tokens and grid, stripping one special token when necessary."""

    if tokens.ndim != 3:
        raise ValueError(f"SigLIP2 image hidden states must be [B,N,C], got {tuple(tokens.shape)}")
    num_tokens = tokens.shape[1]
    try:
        grid_h, grid_w = infer_square_grid(num_tokens)
        return (
            tokens,
            PatchGrid(
                height=grid_h,
                width=grid_w,
                image_height=int(image.shape[-2]),
                image_width=int(image.shape[-1]),
            ),
            0,
        )
    except ValueError:
        if num_tokens <= 1:
            raise
    grid_h, grid_w = infer_square_grid(num_tokens - 1)
    return (
        tokens[:, 1:, :],
        PatchGrid(
            height=grid_h,
            width=grid_w,
            image_height=int(image.shape[-2]),
            image_width=int(image.shape[-1]),
        ),
        1,
    )
