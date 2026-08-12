"""Centralized DGM4 annotation-to-image path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .constants import (
    FACE_ATTRIBUTE_GENERATORS,
    FACE_SWAP_GENERATORS,
    MANIPULATION_GENERATOR_ALIASES,
    ORIGIN_SOURCES,
)


class DGM4PathError(ValueError):
    """Raised when an annotation image path cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedImagePath:
    """Resolved image metadata for a DGM4 annotation path."""

    annotation_path: str
    physical_path: Path
    kind: str
    source: str | None
    generator: str | None


def _parts(annotation_path: str) -> tuple[str, ...]:
    normalized = annotation_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts
    if not parts or parts[0] != "DGM4":
        raise DGM4PathError(
            f"DGM4 annotation path must start with 'DGM4/': {annotation_path!r}"
        )
    if any(part in {"", ".", ".."} for part in parts):
        raise DGM4PathError(f"Malformed DGM4 annotation path: {annotation_path!r}")
    return parts


def resolve_image_path(
    annotation_path: str,
    dataset_root: str | Path,
    *,
    image_root: str | Path | None = None,
    verify_exists: bool = False,
) -> ResolvedImagePath:
    """Resolve a DGM4 JSON image path to the local physical image path."""

    root = Path(dataset_root)
    image_base = Path(image_root) if image_root is not None else root / "images"
    parts = _parts(annotation_path)
    if len(parts) < 4:
        raise DGM4PathError(f"Incomplete DGM4 annotation path: {annotation_path!r}")

    section = parts[1]
    if section == "manipulation":
        if len(parts) != 4:
            raise DGM4PathError(
                "Manipulation paths must be DGM4/manipulation/{method}/{file}.jpg: "
                f"{annotation_path!r}"
            )
        method = parts[2]
        generator = MANIPULATION_GENERATOR_ALIASES.get(method)
        if generator is None:
            raise DGM4PathError(f"Unknown manipulation generator {method!r}")
        if generator not in FACE_SWAP_GENERATORS and generator not in FACE_ATTRIBUTE_GENERATORS:
            raise DGM4PathError(f"Unknown manipulation generator {method!r}")
        physical = image_base / "manipulation" / method / method / parts[3]
        resolved = ResolvedImagePath(
            annotation_path="/".join(parts),
            physical_path=physical,
            kind="manipulation",
            source=None,
            generator=generator,
        )
    elif section == "origin":
        if len(parts) != 5:
            raise DGM4PathError(
                "Origin paths must be DGM4/origin/{source}/{folder}/{file}.jpg: "
                f"{annotation_path!r}"
            )
        source = parts[2]
        if source not in ORIGIN_SOURCES:
            raise DGM4PathError(f"Unknown origin source {source!r}")
        physical = image_base / "origin" / source / source / parts[3] / parts[4]
        resolved = ResolvedImagePath(
            annotation_path="/".join(parts),
            physical_path=physical,
            kind="origin",
            source=source,
            generator=None,
        )
    else:
        raise DGM4PathError(f"Unknown DGM4 image section {section!r}")

    if verify_exists and not resolved.physical_path.exists():
        raise DGM4PathError(
            "Resolved image path does not exist: "
            f"{resolved.annotation_path!r} -> {resolved.physical_path}"
        )
    return resolved


def image_source_or_generator(annotation_path: str) -> str:
    """Return the canonical source/generator token for an annotation path."""

    parts = _parts(annotation_path)
    if len(parts) < 3:
        raise DGM4PathError(f"Incomplete DGM4 annotation path: {annotation_path!r}")
    if parts[1] == "manipulation":
        generator = MANIPULATION_GENERATOR_ALIASES.get(parts[2])
        if generator is None:
            raise DGM4PathError(f"Unknown manipulation generator {parts[2]!r}")
        return generator
    return parts[2]
