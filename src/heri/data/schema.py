"""DGM4 annotation schema validation."""

from __future__ import annotations

from typing import Any, Mapping

from .constants import EXPECTED_SCHEMA_FIELDS
from .labels import UnknownDGM4LabelError, validate_fake_cls


class DGM4SchemaError(ValueError):
    """Raised when a DGM4 annotation row violates the frozen schema."""


def _context(split: str, row_index: int, row: Mapping[str, Any] | None) -> str:
    sample_id = row.get("id", "<missing>") if isinstance(row, Mapping) else "<missing>"
    return f"split={split} row_index={row_index} id={sample_id}"


def _fail(
    *,
    split: str,
    row_index: int,
    row: Mapping[str, Any] | None,
    field: str,
    expected: str,
    actual: Any,
) -> None:
    raise DGM4SchemaError(
        f"DGM4 schema error ({_context(split, row_index, row)}): "
        f"field={field!r} expected={expected}, actual_type={type(actual).__name__}, "
        f"actual={actual!r}"
    )


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, int) for v in value)


def _is_box(value: Any) -> bool:
    return _is_int_list(value) and len(value) == 4


def validate_row(row: Mapping[str, Any], *, split: str, row_index: int) -> None:
    """Validate one DGM4 annotation row."""

    if not isinstance(row, Mapping):
        _fail(split=split, row_index=row_index, row=None, field="<row>", expected="object", actual=row)

    for field in EXPECTED_SCHEMA_FIELDS:
        if field not in row:
            _fail(
                split=split,
                row_index=row_index,
                row=row,
                field=field,
                expected="present field",
                actual="<missing>",
            )

    if not isinstance(row["id"], int):
        _fail(split=split, row_index=row_index, row=row, field="id", expected="int", actual=row["id"])
    if not isinstance(row["image"], str):
        _fail(split=split, row_index=row_index, row=row, field="image", expected="str", actual=row["image"])
    if not isinstance(row["text"], str):
        _fail(split=split, row_index=row_index, row=row, field="text", expected="str", actual=row["text"])
    if not isinstance(row["fake_cls"], str):
        _fail(
            split=split,
            row_index=row_index,
            row=row,
            field="fake_cls",
            expected="str",
            actual=row["fake_cls"],
        )
    try:
        validate_fake_cls(row["fake_cls"])
    except UnknownDGM4LabelError as exc:
        raise DGM4SchemaError(
            f"DGM4 schema error ({_context(split, row_index, row)}): "
            f"field='fake_cls' expected=known label, actual={row['fake_cls']!r}"
        ) from exc

    fake_image_box = row["fake_image_box"]
    if not (fake_image_box == [] or _is_box(fake_image_box)):
        _fail(
            split=split,
            row_index=row_index,
            row=row,
            field="fake_image_box",
            expected="[] or [int,int,int,int]",
            actual=fake_image_box,
        )
    if not _is_int_list(row["fake_text_pos"]):
        _fail(
            split=split,
            row_index=row_index,
            row=row,
            field="fake_text_pos",
            expected="list[int]",
            actual=row["fake_text_pos"],
        )
    mtcnn_boxes = row["mtcnn_boxes"]
    if not isinstance(mtcnn_boxes, list) or not all(_is_box(box) for box in mtcnn_boxes):
        _fail(
            split=split,
            row_index=row_index,
            row=row,
            field="mtcnn_boxes",
            expected="list[[int,int,int,int]]",
            actual=mtcnn_boxes,
        )


def validate_text_positions(words: list[str], positions: list[int], *, context: str) -> None:
    """Validate frozen 0-based word-level fake_text_pos positions."""

    for position in positions:
        if position < 0 or position >= len(words):
            raise DGM4SchemaError(
                f"Invalid fake_text_pos at {context}: position={position}, "
                f"num_words={len(words)}"
            )
