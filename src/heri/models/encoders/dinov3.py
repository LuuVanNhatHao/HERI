"""DINOv3 dense visual encoder wrapper."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from heri.compat.transformers_compat import load_auto_model

from .common import (
    freeze_module,
    infer_square_grid,
    masked_mean,
    normalize_image_tensor,
    pixel_mask_to_patch_mask,
    unfreeze_last_blocks,
)
from ..outputs import DINOv3Output, PatchGrid


class DINOv3Encoder(nn.Module):
    """Production DINOv3 wrapper for dense visual features."""

    def __init__(
        self,
        *,
        model: nn.Module,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        model_name: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)
        self.model_name = model_name

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DINOv3Encoder":
        """Load the configured DINOv3 model."""

        name = str(config["name"])
        runtime = config.get("runtime", {})
        model = load_auto_model(
            name=name,
            stage="DINOv3",
            revision=config.get("revision"),
            local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            torch_dtype=runtime.get("model_dtype"),
            attn_implementation=runtime.get("attn_implementation"),
        )
        return cls(
            model=model,
            image_mean=config["image_mean"],
            image_std=config["image_std"],
            model_name=name,
        )

    def freeze_all(self) -> None:
        """Freeze all DINOv3 parameters."""

        freeze_module(self.model)

    def unfreeze_last_blocks(self, count: int) -> int:
        """Unfreeze final DINOv3 blocks via compatibility introspection."""

        return unfreeze_last_blocks(self.model, count, stage_name="DINOv3")

    def forward(self, *, image: torch.Tensor, pixel_mask: torch.Tensor | None = None) -> DINOv3Output:
        """Encode letterboxed images into patch and global features."""

        pixel_values = normalize_image_tensor(
            image,
            mean=self.image_mean,
            std=self.image_std,
        )
        try:
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
        except TypeError:
            outputs = self.model(pixel_values=pixel_values)
        tokens = _extract_last_hidden_state(outputs, stage="DINOv3")
        if tokens.ndim != 3:
            raise ValueError(f"DINOv3 hidden states must be [B,N,C], got {tuple(tokens.shape)}")

        patch_tokens, grid_h, grid_w, special_token_count = _validated_patch_tokens(tokens)

        patch_mask = pixel_mask_to_patch_mask(
            pixel_mask,
            grid_height=grid_h,
            grid_width=grid_w,
            batch_size=patch_tokens.shape[0],
            device=patch_tokens.device,
        )
        global_feature = _extract_optional_tensor(outputs, "pooler_output")
        if global_feature is None:
            global_feature = (
                tokens[:, 0, :]
                if patch_tokens.shape[1] != tokens.shape[1]
                else masked_mean(patch_tokens, patch_mask)
            )
        return DINOv3Output(
            patch_tokens=patch_tokens,
            patch_mask=patch_mask,
            global_feature=global_feature,
            patch_grid=PatchGrid(
                height=grid_h,
                width=grid_w,
                image_height=int(image.shape[-2]),
                image_width=int(image.shape[-1]),
            ),
            patch_special_token_count=special_token_count,
        )


def _extract_last_hidden_state(outputs: Any, *, stage: str) -> torch.Tensor:
    value = getattr(outputs, "last_hidden_state", None)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(outputs, Mapping):
        value = outputs.get("last_hidden_state")
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError(f"{stage} output missing last_hidden_state")


def _extract_optional_tensor(outputs: Any, name: str) -> torch.Tensor | None:
    value = getattr(outputs, name, None)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(outputs, Mapping):
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _validated_patch_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """Strip prefix special/register tokens until DINO patches form a square grid."""

    num_tokens = int(tokens.shape[1])
    if num_tokens <= 0:
        infer_square_grid(num_tokens)
    max_prefix_special_tokens = min(num_tokens - 1, 16)
    for special_token_count in range(max_prefix_special_tokens + 1):
        patch_count = num_tokens - special_token_count
        try:
            grid_h, grid_w = infer_square_grid(patch_count)
        except ValueError:
            continue
        return tokens[:, special_token_count:, :], grid_h, grid_w, special_token_count
    raise ValueError(
        "Cannot infer DINOv3 square patch grid from "
        f"{num_tokens} tokens after stripping up to {max_prefix_special_tokens} "
        "prefix special/register tokens"
    )
