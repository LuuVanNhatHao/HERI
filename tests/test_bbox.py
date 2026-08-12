import pytest

from src.data.bbox import BBoxError, best_iou_match, iou, transform_box, validate_box
from src.data.transforms import compute_resize_padding


def test_validate_and_iou_identity() -> None:
    box = [1, 2, 11, 12]
    assert validate_box(box, image_size=(20, 20)) == (1.0, 2.0, 11.0, 12.0)
    assert iou(box, box) == 1.0


def test_resize_padding_transform() -> None:
    meta = compute_resize_padding(100, 50, target_size=384)
    assert meta.resized_width == 384
    assert meta.resized_height == 192
    assert meta.pad_top == 96
    transformed = transform_box([10, 10, 30, 30], scale=meta.scale, pad_left=meta.pad_left, pad_top=meta.pad_top)
    assert transformed == pytest.approx((38.4, 134.4, 115.2, 211.2))


def test_best_iou_match() -> None:
    idx, score = best_iou_match([10, 10, 30, 30], [[0, 0, 5, 5], [10, 10, 30, 30]])
    assert idx == 1
    assert score == 1.0


def test_invalid_bbox_handling() -> None:
    with pytest.raises(BBoxError):
        validate_box([10, 10, 5, 5])
