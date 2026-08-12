"""Image resize/padding transforms and matching bbox geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .bbox import Box, transform_box, transform_boxes


@dataclass(frozen=True)
class ImageTransformMeta:
    """Deterministic resize+padding transform metadata."""

    original_width: int
    original_height: int
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    target_width: int
    target_height: int

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-serializable metadata."""

        return asdict(self)


def compute_resize_padding(
    original_width: int,
    original_height: int,
    *,
    target_size: int,
) -> ImageTransformMeta:
    """Compute aspect-ratio preserving resize and symmetric padding."""

    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Invalid image size {original_width}x{original_height}")
    scale = min(target_size / original_width, target_size / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    pad_left = (target_size - resized_width) // 2
    pad_top = (target_size - resized_height) // 2
    pad_right = target_size - resized_width - pad_left
    pad_bottom = target_size - resized_height - pad_top
    return ImageTransformMeta(
        original_width=original_width,
        original_height=original_height,
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        target_width=target_size,
        target_height=target_size,
    )


def preprocess_image(
    image: Image.Image,
    *,
    target_size: int,
    fill: int = 0,
) -> tuple[torch.Tensor, ImageTransformMeta]:
    """Convert an image to RGB, resize with padding, and return CHW tensor."""

    rgb = image.convert("RGB")
    meta = compute_resize_padding(*rgb.size, target_size=target_size)
    resized = rgb.resize((meta.resized_width, meta.resized_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), color=(fill, fill, fill))
    canvas.paste(resized, (meta.pad_left, meta.pad_top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, meta


def transform_box_with_meta(box: Sequence[float | int], meta: ImageTransformMeta) -> Box:
    """Transform one original-coordinate box with image transform metadata."""

    return transform_box(
        box,
        scale=meta.scale,
        pad_left=meta.pad_left,
        pad_top=meta.pad_top,
    )


def transform_boxes_with_meta(
    boxes: Sequence[Sequence[float | int]],
    meta: ImageTransformMeta,
) -> list[Box]:
    """Transform multiple original-coordinate boxes with image transform metadata."""

    return transform_boxes(
        boxes,
        scale=meta.scale,
        pad_left=meta.pad_left,
        pad_top=meta.pad_top,
    )
