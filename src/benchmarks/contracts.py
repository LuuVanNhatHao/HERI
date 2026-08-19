"""Shared data and supervision contracts for binary HERA benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PRIMARY_TASK = "binary_multimodal"
SUPPORTED_DATASETS = frozenset({"multibanfakedetect", "vldbench"})
SUBTYPE_CLASSES = ("non-fake", "clickbait", "misinformation", "rumor")


@dataclass(frozen=True)
class SignalContract:
    """Signals that may legally contribute supervised loss for a benchmark."""

    binary: bool = True
    attribution: bool = False
    image_grounding: bool = False
    text_grounding: bool = False
    subtype: bool = False
    text_label: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SignalContract":
        values = config["task"]["signals"]
        return cls(
            **{name: bool(values.get(name, False)) for name in cls.__dataclass_fields__}
        )

    def as_dict(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in self.__dataclass_fields__}


def normalize_multiban_subtype(value: Any) -> str:
    """Normalize the released MultiBanFakeDetect subtype spelling."""

    normalized = str(value).strip().casefold()
    if normalized not in SUBTYPE_CLASSES:
        raise ValueError(
            f"Unknown MultiBanFakeDetect subtype {value!r}; expected {list(SUBTYPE_CLASSES)}"
        )
    return normalized


def binary_label_from_vldbench(value: Any) -> float:
    """Map the released VLDBench Likely/Unlikely labels without reinterpreting them."""

    normalized = str(value).strip().casefold()
    if normalized == "likely":
        return 1.0
    if normalized == "unlikely":
        return 0.0
    raise ValueError(f"Unknown VLDBench label: {value!r}")
