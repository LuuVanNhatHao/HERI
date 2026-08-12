"""Centralized DGM4 label conversion.

No other module should parse ``fake_cls`` directly.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .constants import LABEL_ORDER


class UnknownDGM4LabelError(ValueError):
    """Raised when a DGM4 ``fake_cls`` value is not canonical."""


LABEL_TO_ATOMIC: dict[str, tuple[int, int, int, int]] = {
    "orig": (0, 0, 0, 0),
    "face_swap": (1, 0, 0, 0),
    "face_attribute": (0, 1, 0, 0),
    "text_swap": (0, 0, 1, 0),
    "text_attribute": (0, 0, 0, 1),
    "face_swap&text_swap": (1, 0, 1, 0),
    "face_attribute&text_swap": (0, 1, 1, 0),
    "face_swap&text_attribute": (1, 0, 0, 1),
    "face_attribute&text_attribute": (0, 1, 0, 1),
}


def validate_fake_cls(fake_cls: str) -> None:
    """Validate a raw DGM4 class string."""

    if fake_cls not in LABEL_TO_ATOMIC:
        raise UnknownDGM4LabelError(f"Unknown DGM4 fake_cls: {fake_cls!r}")


def atomic_target(fake_cls: str) -> tuple[int, int, int, int]:
    """Return the canonical [FS, FA, TS, TA] target."""

    validate_fake_cls(fake_cls)
    return LABEL_TO_ATOMIC[fake_cls]


def binary_target(fake_cls: str) -> int:
    """Return binary fake target: orig -> 0, all manipulations -> 1."""

    validate_fake_cls(fake_cls)
    return 0 if fake_cls == "orig" else 1


def has_image_manipulation(fake_cls: str) -> bool:
    """Return whether the label contains FS or FA."""

    fs, fa, _, _ = atomic_target(fake_cls)
    return bool(fs or fa)


def has_text_manipulation(fake_cls: str) -> bool:
    """Return whether the label contains TS or TA."""

    _, _, ts, ta = atomic_target(fake_cls)
    return bool(ts or ta)


def modality_group(fake_cls: str) -> str:
    """Return a compact manipulation-modality group."""

    if fake_cls == "orig":
        return "real"
    image = has_image_manipulation(fake_cls)
    text = has_text_manipulation(fake_cls)
    if image and text:
        return "mixed"
    if image:
        return "image_only"
    if text:
        return "text_only"
    raise UnknownDGM4LabelError(f"Unable to determine modality for {fake_cls!r}")


def atomic_counts(fake_classes: Iterable[str]) -> Counter[str]:
    """Count atomic labels in canonical label order."""

    counts: Counter[str] = Counter({name: 0 for name in LABEL_ORDER})
    for fake_cls in fake_classes:
        for name, value in zip(LABEL_ORDER, atomic_target(fake_cls)):
            counts[name] += int(value)
    return counts
