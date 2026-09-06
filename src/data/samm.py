"""SAMM-20K adapter normalized to HERA's dataset-independent batch contract."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .bbox import BBoxError, clip_box
from .constants import ABSENT_BOX
from .dataset import select_smoke_rows
from .tokenizer_alignment import align_word_positions_to_subwords
from .transforms import image_content_mask, preprocess_image, transform_box_with_meta


SAMM_HYPOTHESIS_ORDER: tuple[str, ...] = ("SWAP", "ATTRIBUTE")
SAMM_CLASS_ORDER: tuple[str, ...] = (
    "orig",
    "swap_manipulation",
    "attribute_manipulation",
)
SAMM_LABEL_TO_TARGET: dict[str, tuple[int, int]] = {
    "orig": (0, 0),
    "swap_manipulation": (1, 0),
    "attribute_manipulation": (0, 1),
}
SAMM_LABEL_TO_FOLDER = {
    "orig": "orig_output",
    "swap_manipulation": "swap_jpg",
    "attribute_manipulation": "emotion_jpg",
}
SAMM_EXPECTED_SPLIT_ROWS = {"train": 20_000, "val": 20_000, "test": 50_000}
SAMM_EXPECTED_SHA256 = {
    "train": "86ddcace282a029633446ca382523a3867ac2238b846182dda956113283dcbd5",
    "val": "dd5663c7775024bc14741ccb58efba37a0b9033c926365f960756692ea041df0",
    "test": "2c4d43a67b00ad85805c0ece348433d0a485f138b28344cca002d66957d14552",
}


class SAMMSchemaError(ValueError):
    """Raised when a SAMM annotation violates the supported schema."""


class SAMMPathError(ValueError):
    """Raised when a SAMM image path is unsafe or inconsistent with its class."""


def load_samm_annotations(path: str | Path, *, split: str) -> list[dict[str, Any]]:
    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise SAMMSchemaError(f"SAMM {split} annotation file must contain a list: {path}")
    return rows


def validate_samm_row(row: Mapping[str, Any], *, split: str, row_index: int) -> None:
    context = f"split={split} row_index={row_index} id={row.get('id', '<missing>') if isinstance(row, Mapping) else '<missing>'}"
    if not isinstance(row, Mapping):
        raise SAMMSchemaError(f"SAMM schema error ({context}): row must be an object")
    for field in ("id", "image", "text", "fake_cls", "fake_text_pos"):
        if field not in row:
            raise SAMMSchemaError(f"SAMM schema error ({context}): missing field {field!r}")
    if not isinstance(row["id"], int):
        raise SAMMSchemaError(f"SAMM schema error ({context}): id must be int")
    if not isinstance(row["image"], str) or not isinstance(row["text"], str):
        raise SAMMSchemaError(f"SAMM schema error ({context}): image/text must be strings")
    label = row["fake_cls"]
    if label not in SAMM_LABEL_TO_TARGET:
        raise SAMMSchemaError(f"SAMM schema error ({context}): unknown fake_cls={label!r}")
    positions = row["fake_text_pos"]
    if not isinstance(positions, list) or not all(isinstance(value, int) for value in positions):
        raise SAMMSchemaError(f"SAMM schema error ({context}): fake_text_pos must be list[int]")
    box = row.get("fake_image_box", [])
    if box != [] and (
        not isinstance(box, list)
        or len(box) != 4
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in box)
    ):
        raise SAMMSchemaError(f"SAMM schema error ({context}): fake_image_box must be absent, [], or four numbers")
    if label == "orig" and (box or positions):
        raise SAMMSchemaError(f"SAMM schema error ({context}): orig rows must have empty grounding annotations")


def resolve_samm_image_path(
    annotation_path: str,
    dataset_root: str | Path,
    *,
    image_root: str | Path | None = None,
    fake_cls: str | None = None,
    verify_exists: bool = False,
) -> Path:
    normalized = annotation_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts
    if len(parts) != 3 or parts[0] != "images" or any(part in {"", ".", ".."} for part in parts):
        raise SAMMPathError(
            "SAMM annotation path must be images/{orig_output|swap_jpg|emotion_jpg}/{file}: "
            f"{annotation_path!r}"
        )
    if fake_cls is not None:
        expected_folder = SAMM_LABEL_TO_FOLDER.get(fake_cls)
        if expected_folder is None or parts[1] != expected_folder:
            raise SAMMPathError(
                f"SAMM label/path mismatch: fake_cls={fake_cls!r}, path={annotation_path!r}"
            )
    base = Path(image_root) if image_root is not None else Path(dataset_root) / "images"
    physical = base / parts[1] / parts[2]
    if verify_exists and not physical.is_file():
        raise SAMMPathError(f"SAMM image does not exist: {annotation_path!r} -> {physical}")
    return physical


class SAMMDataset(Dataset[dict[str, Any]]):
    """SAMM dataset with coupled hypotheses and annotation-availability masks."""

    def __init__(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        split: str,
        dataset_root: str | Path,
        image_size: int,
        image_root: str | Path | None = None,
        tokenizer: Any | None = None,
        tokenizer_max_length: int | None = None,
        tokenizer_ignore_index: int = -100,
        verify_schema: bool = False,
        verify_image_exists: bool = True,
        progress: bool = False,
        clip_out_of_bounds_boxes: bool = True,
        mask_missing_image_boxes: bool = True,
        mask_invalid_text_positions: bool = True,
        jpeg_draft: bool = True,
    ) -> None:
        self.rows = list(rows)
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.image_root = Path(image_root) if image_root is not None else self.dataset_root / "images"
        self.image_size = int(image_size)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer_ignore_index = int(tokenizer_ignore_index)
        self.verify_image_exists = bool(verify_image_exists)
        self.clip_out_of_bounds_boxes = bool(clip_out_of_bounds_boxes)
        self.mask_missing_image_boxes = bool(mask_missing_image_boxes)
        self.mask_invalid_text_positions = bool(mask_invalid_text_positions)
        self.jpeg_draft = bool(jpeg_draft)
        if verify_schema:
            iterator = tqdm(
                enumerate(self.rows), total=len(self.rows),
                desc=f"SAMM {split} dataset schema", disable=not progress,
            )
            for index, row in iterator:
                validate_samm_row(row, split=split, row_index=index)

    @classmethod
    def from_annotation_file(cls, *, annotation_path: str | Path, max_samples: int | None = None,
                             samples_per_class: int | None = None, **kwargs: Any) -> "SAMMDataset":
        rows = load_samm_annotations(annotation_path, split=str(kwargs["split"]))
        rows = select_smoke_rows(rows, max_samples=max_samples, samples_per_class=samples_per_class)
        return cls(rows=rows, **kwargs)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        validate_samm_row(row, split=self.split, row_index=index)
        label = str(row["fake_cls"])
        physical = resolve_samm_image_path(
            row["image"], self.dataset_root, image_root=self.image_root,
            fake_cls=label, verify_exists=self.verify_image_exists,
        )
        with Image.open(physical) as image:
            original_size = image.size
            if self.jpeg_draft and image.format == "JPEG":
                image.draft("RGB", (self.image_size * 2, self.image_size * 2))
            image_tensor, transform_meta = preprocess_image(
                image, target_size=self.image_size, geometry_size=original_size
            )

        manipulated = label != "orig"
        box_target = torch.tensor(ABSENT_BOX, dtype=torch.float32)
        image_annotation_available = not manipulated
        raw_box = row.get("fake_image_box", [])
        if manipulated and raw_box:
            try:
                if self.clip_out_of_bounds_boxes:
                    box = clip_box(raw_box, width=original_size[0], height=original_size[1])
                else:
                    from .bbox import validate_box
                    box = validate_box(raw_box, image_size=original_size, field_name="fake_image_box")
                box_target = torch.tensor(
                    transform_box_with_meta(box, transform_meta), dtype=torch.float32
                )
                image_annotation_available = True
            except BBoxError:
                if not self.mask_missing_image_boxes:
                    raise
                image_annotation_available = False
        elif manipulated and not self.mask_missing_image_boxes:
            raise SAMMSchemaError(
                f"Missing SAMM fake_image_box at split={self.split} index={index} id={row['id']}"
            )

        text = str(row["text"])
        words = text.split()
        raw_positions = list(row["fake_text_pos"])
        positions_valid = all(0 <= position < len(words) for position in raw_positions)
        if not positions_valid and not self.mask_invalid_text_positions:
            bad = [position for position in raw_positions if position < 0 or position >= len(words)]
            raise SAMMSchemaError(
                f"Invalid SAMM fake_text_pos at split={self.split} index={index} id={row['id']}: {bad}"
            )
        safe_positions = raw_positions if positions_valid else []
        text_annotation_available = (not manipulated) or (positions_valid and bool(raw_positions))
        alignment = align_word_positions_to_subwords(
            words=words,
            fake_text_pos=safe_positions,
            tokenizer=self.tokenizer,
            max_length=self.tokenizer_max_length,
            ignore_index=self.tokenizer_ignore_index,
        )
        target = SAMM_LABEL_TO_TARGET[label]
        item: dict[str, Any] = {
            "id": row["id"], "split": self.split, "dataset_name": "SAMM",
            "image": image_tensor, "image_content_mask": image_content_mask(transform_meta),
            "image_path": str(physical), "annotation_image_path": row["image"],
            "image_transform_meta": transform_meta.to_dict(), "text": text, "words": words,
            "binary_target": torch.tensor(int(manipulated), dtype=torch.float32),
            "atomic_target": torch.tensor(target, dtype=torch.float32),
            "fake_image_box_target": box_target,
            "has_image_manipulation": torch.tensor(manipulated, dtype=torch.bool),
            "image_grounding_available": torch.tensor(image_annotation_available, dtype=torch.bool),
            "fake_text_pos": torch.tensor(safe_positions, dtype=torch.long),
            "text_grounding_target": torch.tensor(alignment.labels, dtype=torch.long),
            "text_grounding_mask": torch.tensor(alignment.mask, dtype=torch.bool),
            "text_word_ids": torch.tensor(
                [-1 if word_id is None else int(word_id) for word_id in alignment.word_ids],
                dtype=torch.long,
            ),
            "has_text_manipulation": torch.tensor(manipulated, dtype=torch.bool),
            "text_grounding_available": torch.tensor(text_annotation_available, dtype=torch.bool),
            "fake_cls": label,
            "generator": None if label == "orig" else label,
            "image_source": "orig" if label == "orig" else None,
            "class_target": SAMM_CLASS_ORDER.index(label),
            "class_names": list(SAMM_CLASS_ORDER),
            "hypothesis_names": list(SAMM_HYPOTHESIS_ORDER),
        }
        encoding = alignment.encoding
        if encoding is not None and "input_ids" in encoding:
            item["text_input_ids"] = torch.tensor(encoding["input_ids"], dtype=torch.long)
            item["text_attention_mask"] = torch.tensor(
                encoding.get("attention_mask", [1] * len(encoding["input_ids"])), dtype=torch.bool
            )
        else:
            item["text_input_ids"] = torch.arange(len(words), dtype=torch.long)
            item["text_attention_mask"] = torch.ones(len(words), dtype=torch.bool)
        return item


def samm_class_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("fake_cls")) for row in rows)
