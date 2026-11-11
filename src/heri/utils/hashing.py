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


def source_tree_sha256(repo_root: str | Path) -> str:
    """Identify executable source, configurations and dependency constraints."""
    root = Path(repo_root)
    files = [
        *root.joinpath("src").rglob("*.py"),
        *root.joinpath("scripts").glob("*.py"),
        *root.joinpath("config").rglob("*.yaml"),
        *root.joinpath("requirements").glob("*.txt"),
        root / "pyproject.toml",
        root / "requirements.txt",
    ]
    digest = hashlib.sha256()
    for path in sorted(path for path in files if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()
