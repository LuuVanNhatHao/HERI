"""Byte-exact annotation recipe from HERI Appendix A, independent of images."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Mapping, Sequence


def reconstruct_annotation_bytes(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    train_size: int = 20_000,
    seed: int = 777,
) -> tuple[bytes, bytes]:
    """Return compact UTF-8 annotations and decimal-ID/LF ordered manifest.

    Training samples source IDs in their original order, then sorts selected
    integer IDs. Validation/test preserve source row order. This function is
    also usable with a few synthetic rows to verify the recipe locally.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown SAMM split: {split}")
    ids = [row["id"] for row in source_rows]
    if any(type(value) is not int for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Source SAMM IDs must be unique integers")
    if split == "train":
        if train_size < 1 or train_size > len(ids):
            raise ValueError("Training subset size exceeds the source population")
        selected = sorted(random.Random(seed).sample(ids, train_size))
        by_id = {row["id"]: row for row in source_rows}
        rows = [dict(by_id[value]) for value in selected]
    else:
        rows = [dict(row) for row in source_rows]
    for row in rows:
        image = row["image"]
        if not isinstance(image, str) or image.startswith("images/"):
            raise ValueError(
                "Reconstruction requires original unprefixed SAMM image paths"
            )
        row["image"] = "images/" + image
    annotations = json.dumps(
        rows, ensure_ascii=False, separators=(",", ":"), indent=None
    ).encode("utf-8")
    ordered_ids = "".join(f"{row['id']}\n" for row in rows).encode("utf-8")
    return annotations, ordered_ids


def annotation_fingerprint(annotations: bytes, ordered_ids: bytes) -> dict[str, Any]:
    rows = json.loads(annotations)
    return {
        "rows": len(rows),
        "sha256": hashlib.sha256(annotations).hexdigest(),
        "ordered_id_manifest_sha256": hashlib.sha256(ordered_ids).hexdigest(),
    }
