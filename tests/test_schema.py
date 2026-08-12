import pytest

from src.data.schema import DGM4SchemaError, validate_row


def valid_row() -> dict:
    return {
        "id": 1,
        "image": "DGM4/origin/bbc/0000/001.jpg",
        "text": "Alice meets Bob",
        "fake_cls": "orig",
        "fake_image_box": [],
        "fake_text_pos": [],
        "mtcnn_boxes": [[10, 10, 30, 30]],
    }


def test_valid_row() -> None:
    validate_row(valid_row(), split="train", row_index=0)


def test_missing_field() -> None:
    row = valid_row()
    row.pop("text")
    with pytest.raises(DGM4SchemaError):
        validate_row(row, split="train", row_index=0)


def test_invalid_field_type() -> None:
    row = valid_row()
    row["id"] = "1"
    with pytest.raises(DGM4SchemaError):
        validate_row(row, split="train", row_index=0)


def test_invalid_fake_cls() -> None:
    row = valid_row()
    row["fake_cls"] = "new_label"
    with pytest.raises(DGM4SchemaError):
        validate_row(row, split="train", row_index=0)
