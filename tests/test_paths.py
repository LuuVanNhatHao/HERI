from pathlib import Path

import pytest

from src.data.paths import DGM4PathError, resolve_image_path


def test_origin_path_resolution(tiny_dgm4_root: Path) -> None:
    resolved = resolve_image_path(
        "DGM4/origin/bbc/0000/001.jpg",
        tiny_dgm4_root,
        verify_exists=True,
    )
    assert resolved.kind == "origin"
    assert resolved.source == "bbc"
    assert resolved.physical_path.name == "001.jpg"


@pytest.mark.parametrize(
    ("annotation", "generator"),
    [
        ("DGM4/manipulation/infoswap/1-2-infoswap.jpg", "InfoSwap"),
        ("DGM4/manipulation/simswap/2-3-simswap.jpg", "SimSwap"),
        ("DGM4/manipulation/HFGI/3-HFGI.jpg", "HFGI"),
        ("DGM4/manipulation/StyleCLIP/4-StyleCLIP.jpg", "StyleCLIP"),
    ],
)
def test_manipulation_path_resolution(tiny_dgm4_root: Path, annotation: str, generator: str) -> None:
    resolved = resolve_image_path(annotation, tiny_dgm4_root, verify_exists=True)
    assert resolved.kind == "manipulation"
    assert resolved.generator == generator


def test_malformed_path_rejection(tiny_dgm4_root: Path) -> None:
    with pytest.raises(DGM4PathError):
        resolve_image_path("../bad.jpg", tiny_dgm4_root)
