from __future__ import annotations

from pathlib import Path


def _requirement_names() -> set[str]:
    names: set[str] = set()
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = stripped.split(";", maxsplit=1)[0]
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            name = name.split(separator, maxsplit=1)[0]
        names.add(name.strip().lower())
    return names


def test_transformers_tokenizer_optional_dependencies_are_declared() -> None:
    requirements = _requirement_names()
    assert "sentencepiece" in requirements
    assert "tiktoken" in requirements
