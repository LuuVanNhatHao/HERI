"""DINOv3 dense visual encoder wrapper."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from src.compat.transformers_compat import load_auto_model

from .common import freeze_module, infer_square_grid, masked_mean, normalize_image_tensor, unfreeze_last_blocks
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

    def forward(self, *, image: torch.Tensor) -> DINOv3Output:
        """Encode Phase-1 letterboxed images into patch and global features."""

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

        patch_tokens = tokens
        num_tokens = tokens.shape[1]
        try:
            grid_h, grid_w = infer_square_grid(num_tokens)
        except ValueError:
            grid_h, grid_w = infer_square_grid(num_tokens - 1)
            patch_tokens = tokens[:, 1:, :]

        patch_mask = torch.ones(
            patch_tokens.shape[:2],
            dtype=torch.bool,
            device=patch_tokens.device,
        )
        global_feature = _extract_optional_tensor(outputs, "pooler_output")
        if global_feature is None:
            global_feature = tokens[:, 0, :] if patch_tokens.shape[1] != tokens.shape[1] else masked_mean(tokens, None)
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
