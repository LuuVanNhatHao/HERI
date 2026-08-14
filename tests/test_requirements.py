from __future__ import annotations

from pathlib import Path

import pytest

from src.compat import transformers_compat
from src.compat.transformers_compat import TransformersCompatibilityError


def _requirement_names() -> set[str]:
    names: set[str] = set()
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "--")):
            continue
        name = stripped.split(";", maxsplit=1)[0]
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            name = name.split(separator, maxsplit=1)[0]
        names.add(name.strip().lower())
    return names


def _requirement_line(package_name: str) -> str:
    normalized = package_name.lower()
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(normalized):
            return stripped
    raise AssertionError(f"Missing requirement for {package_name}")


def test_transformers_tokenizer_optional_dependencies_are_declared() -> None:
    requirements = _requirement_names()
    assert "sentencepiece" in requirements
    assert "tiktoken" in requirements
    assert "protobuf" in requirements


def test_transformers_is_pinned_to_validated_last_four_x_patch() -> None:
    requirement = _requirement_line("transformers")
    assert requirement == "transformers==4.57.6"


def test_blackwell_cuda_pytorch_stack_is_pinned() -> None:
    contents = Path("requirements.txt").read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in contents
    assert _requirement_line("torch") == "torch==2.8.0+cu128"
    assert _requirement_line("torchvision") == "torchvision==0.23.0+cu128"
    assert _requirement_line("torch-geometric") == "torch-geometric==2.7.0"


def test_huggingface_hub_major_version_matches_transformers_four() -> None:
    requirement = _requirement_line("huggingface-hub")
    assert ">=0.34" in requirement
    assert "<1.0" in requirement


def test_transformers_major_five_fails_with_explicit_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transformers_compat, "_transformers_version", lambda: "5.15.0")
    with pytest.raises(TransformersCompatibilityError, match="unsupported transformers version"):
        transformers_compat._ensure_supported_transformers_version("AutoTokenizer.from_pretrained")
