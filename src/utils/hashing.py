"""Hashing helpers with tqdm progress."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tqdm.auto import tqdm


def sha256_file(path: str | Path, *, progress: bool = True, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for a file."""

    file_path = Path(path)
    total = file_path.stat().st_size
    digest = hashlib.sha256()
    iterator = tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"SHA-256 {file_path.name}",
        disable=not progress,
    )
    with file_path.open("rb") as handle, iterator:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            iterator.update(len(chunk))
    return digest.hexdigest()
