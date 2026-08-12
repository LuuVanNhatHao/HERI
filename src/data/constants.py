"""Frozen DGM4 constants from the Phase 0 Constitution."""

from __future__ import annotations

from dataclasses import dataclass


SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

EXPECTED_SPLIT_ROWS: dict[str, int] = {
    "train": 208_184,
    "val": 22_126,
    "test": 50_705,
}

EXPECTED_TOTAL_ROWS = 281_015
EXPECTED_UNIQUE_PHYSICAL_IMAGES = 206_362
EXPECTED_REAL_ROWS = 128_441
EXPECTED_MANIPULATED_ROWS = 152_574
EXPECTED_UNIQUE_PRISTINE_IDS = 77_426

EXPECTED_ANNOTATION_SHA256: dict[str, str] = {
    "train": "dafd4fe101d3de63b433759fae79b14b465294f7be73f26c9f3bb13b8de87704",
    "val": "d4377ecc19cb6528e36c77401482cbfffaa57977faee2c6124e3b1bdda045520",
    "test": "b58b29c4cf0f40cdf791189181ad57821f9359e358e93683cc702db3a41c0a18",
}

LABEL_ORDER: tuple[str, ...] = ("FS", "FA", "TS", "TA")

EXPECTED_SCHEMA_FIELDS: tuple[str, ...] = (
    "id",
    "image",
    "text",
    "fake_cls",
    "fake_image_box",
    "fake_text_pos",
    "mtcnn_boxes",
)

MANIPULATION_GENERATOR_ALIASES: dict[str, str] = {
    "infoswap": "InfoSwap",
    "simswap": "SimSwap",
    "InfoSwap": "InfoSwap",
    "SimSwap": "SimSwap",
    "HFGI": "HFGI",
    "StyleCLIP": "StyleCLIP",
}
FACE_SWAP_GENERATORS: frozenset[str] = frozenset({"InfoSwap", "SimSwap"})
FACE_ATTRIBUTE_GENERATORS: frozenset[str] = frozenset({"HFGI", "StyleCLIP"})
ORIGIN_SOURCES: frozenset[str] = frozenset(
    {"bbc", "guardian", "usa_today", "washington_post"}
)

EXPECTED_MANIPULATED_IMAGE_ROWS = 123_133
EXPECTED_TEXT_MANIPULATED_ROWS = 62_134
EXPECTED_TEXT_ONLY_ROWS = 29_441
EXPECTED_FAKE_BOX_MTCNN_EXACT_MATCHES = 123_133

DEFAULT_IMAGE_SIZE = 384
ABSENT_BOX = (0.0, 0.0, 0.0, 0.0)
ABSENT_INDEX = -1


@dataclass(frozen=True)
class SplitPaths:
    """Annotation file paths relative to the repository root."""

    train: str = "data/DGM4/train.json"
    val: str = "data/DGM4/val.json"
    test: str = "data/DGM4/test.json"


SPLIT_PATHS = SplitPaths()
