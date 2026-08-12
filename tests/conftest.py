from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


def make_image(path: Path, size: tuple[int, int] = (100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 80, 40)).save(path)


@pytest.fixture
def tiny_dgm4_root(tmp_path: Path) -> Path:
    root = tmp_path / "DGM4"
    make_image(root / "images" / "origin" / "bbc" / "bbc" / "0000" / "001.jpg")
    make_image(root / "images" / "origin" / "bbc" / "bbc" / "0000" / "002.jpg")
    make_image(root / "images" / "manipulation" / "infoswap" / "infoswap" / "1-2-infoswap.jpg")
    make_image(root / "images" / "manipulation" / "simswap" / "simswap" / "2-3-simswap.jpg")
    make_image(root / "images" / "manipulation" / "HFGI" / "HFGI" / "3-HFGI.jpg")
    make_image(root / "images" / "manipulation" / "StyleCLIP" / "StyleCLIP" / "4-StyleCLIP.jpg")
    return root


@pytest.fixture
def sample_rows() -> list[dict]:
    return [
        {
            "id": 1,
            "image": "DGM4/origin/bbc/0000/001.jpg",
            "text": "Alice meets Bob",
            "fake_cls": "orig",
            "fake_image_box": [],
            "fake_text_pos": [],
            "mtcnn_boxes": [[10, 10, 30, 30]],
        },
        {
            "id": 2,
            "image": "DGM4/manipulation/infoswap/1-2-infoswap.jpg",
            "text": "Alice meets Bob",
            "fake_cls": "face_swap",
            "fake_image_box": [10, 10, 30, 30],
            "fake_text_pos": [],
            "mtcnn_boxes": [[10, 10, 30, 30], [50, 10, 70, 30]],
        },
        {
            "id": 3,
            "image": "DGM4/origin/bbc/0000/002.jpg",
            "text": "Alice meets Bob",
            "fake_cls": "text_swap",
            "fake_image_box": [],
            "fake_text_pos": [0, 2],
            "mtcnn_boxes": [[10, 10, 30, 30]],
        },
        {
            "id": 4,
            "image": "DGM4/manipulation/HFGI/3-HFGI.jpg",
            "text": "Alice meets Bob",
            "fake_cls": "face_attribute&text_attribute",
            "fake_image_box": [10, 10, 30, 30],
            "fake_text_pos": [1],
            "mtcnn_boxes": [[10, 10, 30, 30]],
        },
    ]


def write_split(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")
