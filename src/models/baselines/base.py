"""Shared utilities for B0/B1 controlled baselines."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.models.encoders.deberta import ContextTextEncoder
from src.models.encoders.dinov3 import DINOv3Encoder
from src.models.encoders.siglip2 import SigLIP2Encoder
from src.models.outputs import BackboneOutputs


class BaselineBackboneMixin:
    """Shared backbone handling for production baselines."""

    siglip2: SigLIP2Encoder | None
    dinov3: DINOv3Encoder | None
    deberta: ContextTextEncoder | None

    def _encode_backbones(
        self,
        batch: Mapping[str, Any],
        features: BackboneOutputs | None,
    ) -> BackboneOutputs:
        if features is not None:
            return features
        if self.siglip2 is None or self.dinov3 is None or self.deberta is None:
            raise RuntimeError(
                "BackboneOutputs must be provided unless real pretrained encoders are attached"
            )
        image = batch["image"]
        pixel_mask = batch.get("image_content_mask")
        return BackboneOutputs(
            siglip2=self.siglip2(
                image=image,
                texts=list(batch["text"]),
                words=list(batch["words"]) if "words" in batch else None,
                pixel_mask=pixel_mask,
            ),
            dinov3=self.dinov3(image=image, pixel_mask=pixel_mask),
            deberta=self.deberta(
                input_ids=batch["text_input_ids"],
                attention_mask=batch["text_attention_mask"],
            ),
        )

    def backbone_modules(self) -> dict[str, nn.Module]:
        """Return attached backbone modules by optimizer-group name."""

        modules: dict[str, nn.Module] = {}
        if self.siglip2 is not None:
            modules["siglip2"] = self.siglip2
        if self.dinov3 is not None:
            modules["dinov3"] = self.dinov3
        if self.deberta is not None:
            modules["deberta"] = self.deberta
        return modules

    def new_modules(self) -> list[nn.Module]:
        """Return non-backbone trainable modules."""

        backbone_ids = {id(module) for module in self.backbone_modules().values()}
        return [
            module
            for module in self.children()
            if id(module) not in backbone_ids
        ]


def finite_or_fail(output_tensors: Mapping[str, torch.Tensor]) -> None:
    """Fail fast if a baseline produces non-finite outputs."""

    for name, tensor in output_tensors.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite baseline output: {name}")
