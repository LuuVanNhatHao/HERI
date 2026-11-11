from pathlib import Path

from heri.data.dataset import DGM4Dataset


def test_dataset_contract_for_core_sample_types(
    tiny_dgm4_root: Path, sample_rows: list[dict]
) -> None:
    dataset = DGM4Dataset(
        rows=sample_rows,
        split="train",
        dataset_root=tiny_dgm4_root,
        image_size=384,
        verify_image_exists=True,
    )
    orig = dataset[0]
    image_only = dataset[1]
    text_only = dataset[2]
    mixed = dataset[3]
    assert orig["binary_target"].item() == 0
    assert image_only["has_image_manipulation"].item()
    assert text_only["has_text_manipulation"].item()
    assert not text_only["has_image_manipulation"].item()
    assert mixed["atomic_target"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert "candidate_face_boxes" not in image_only
    assert "candidate_face_mask" not in image_only
    assert "manipulated_face_target_index" not in image_only
    assert "fake_image_box_target" in image_only
