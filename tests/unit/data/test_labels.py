from heri.data.labels import (
    atomic_target,
    binary_target,
    has_image_manipulation,
    has_text_manipulation,
)


def test_all_nine_fake_cls_mappings() -> None:
    expected = {
        "orig": (0, 0, 0, 0),
        "face_swap": (1, 0, 0, 0),
        "face_attribute": (0, 1, 0, 0),
        "text_swap": (0, 0, 1, 0),
        "text_attribute": (0, 0, 0, 1),
        "face_swap&text_swap": (1, 0, 1, 0),
        "face_attribute&text_swap": (0, 1, 1, 0),
        "face_swap&text_attribute": (1, 0, 0, 1),
        "face_attribute&text_attribute": (0, 1, 0, 1),
    }
    for label, target in expected.items():
        assert atomic_target(label) == target
        assert binary_target(label) == (0 if label == "orig" else 1)


def test_modality_helpers() -> None:
    assert has_image_manipulation("face_swap")
    assert not has_image_manipulation("text_swap")
    assert has_text_manipulation("text_attribute")
    assert not has_text_manipulation("face_attribute")
