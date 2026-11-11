"""Bounding-box validation and geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


class BBoxError(ValueError):
    """Raised for malformed or invalid bounding boxes."""


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageSize:
    """Image size in pixels."""

    width: int
    height: int


def validate_box(
    box: Sequence[float | int],
    *,
    image_size: ImageSize | tuple[int, int] | None = None,
    field_name: str = "bbox",
) -> Box:
    """Validate and return a [x1, y1, x2, y2] box."""

    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)):
        raise BBoxError(f"{field_name} must be a sequence of four numbers")
    if len(box) != 4:
        raise BBoxError(f"{field_name} must contain four values, got {len(box)}")
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError) as exc:
        raise BBoxError(f"{field_name} contains non-numeric values: {box!r}") from exc
    if not (x1 < x2 and y1 < y2):
        raise BBoxError(f"{field_name} must have positive area: {box!r}")
    if image_size is not None:
        if isinstance(image_size, ImageSize):
            width, height = image_size.width, image_size.height
        else:
            width, height = image_size
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise BBoxError(
                f"{field_name} is out of bounds for {width}x{height}: {box!r}"
            )
    return (x1, y1, x2, y2)


def validate_boxes(
    boxes: Iterable[Sequence[float | int]],
    *,
    image_size: ImageSize | tuple[int, int] | None = None,
    field_name: str = "boxes",
) -> list[Box]:
    """Validate a sequence of boxes."""

    return [
        validate_box(box, image_size=image_size, field_name=f"{field_name}[{idx}]")
        for idx, box in enumerate(boxes)
    ]


def area(box: Sequence[float | int]) -> float:
    """Return box area."""

    x1, y1, x2, y2 = validate_box(box)
    return (x2 - x1) * (y2 - y1)


def iou(box_a: Sequence[float | int], box_b: Sequence[float | int]) -> float:
    """Return intersection-over-union for two boxes."""

    ax1, ay1, ax2, ay2 = validate_box(box_a, field_name="box_a")
    bx1, by1, bx2, by2 = validate_box(box_b, field_name="box_b")
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = area((ax1, ay1, ax2, ay2)) + area((bx1, by1, bx2, by2)) - inter
    return 0.0 if union <= 0 else inter / union


def best_iou_match(
    target_box: Sequence[float | int],
    candidate_boxes: Sequence[Sequence[float | int]],
) -> tuple[int, float]:
    """Return the index and IoU of the best candidate box."""

    if not candidate_boxes:
        return -1, 0.0
    scores = [iou(target_box, candidate) for candidate in candidate_boxes]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return best_index, scores[best_index]


def transform_box(
    box: Sequence[float | int],
    *,
    scale: float,
    pad_left: float,
    pad_top: float,
) -> Box:
    """Apply resize+padding geometry to a box."""

    x1, y1, x2, y2 = validate_box(box)
    return (
        x1 * scale + pad_left,
        y1 * scale + pad_top,
        x2 * scale + pad_left,
        y2 * scale + pad_top,
    )


def transform_boxes(
    boxes: Iterable[Sequence[float | int]],
    *,
    scale: float,
    pad_left: float,
    pad_top: float,
) -> list[Box]:
    """Apply resize+padding geometry to multiple boxes."""

    return [
        transform_box(box, scale=scale, pad_left=pad_left, pad_top=pad_top)
        for box in boxes
    ]


def clip_box(
    box: Sequence[float | int],
    *,
    width: int,
    height: int,
) -> Box:
    """Explicitly clip a box to image bounds.

    This helper is intentionally not used by canonical preflight repair.
    """

    x1, y1, x2, y2 = (float(v) for v in box)
    clipped = (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )
    return validate_box(clipped, field_name="clipped_bbox")
