from pathlib import Path

from src.data.collate import dgm4_collate
from src.data.dataset import DGM4Dataset


def test_collate_variable_faces_and_text(tiny_dgm4_root: Path, sample_rows: list[dict]) -> None:
    dataset = DGM4Dataset(
        rows=sample_rows,
        split="train",
        dataset_root=tiny_dgm4_root,
        image_size=384,
        verify_image_exists=True,
    )
    batch = dgm4_collate([dataset[0], dataset[1], dataset[2]])
    assert batch["image"].shape == (3, 3, 384, 384)
    assert "candidate_face_boxes" not in batch
    assert "candidate_face_mask" not in batch
    assert "manipulated_face_target_index" not in batch
    assert batch["text_grounding_target"].shape == batch["text_grounding_mask"].shape
    assert batch["text_word_ids"].shape == batch["text_grounding_target"].shape
    assert batch["atomic_target"].shape == (3, 4)
