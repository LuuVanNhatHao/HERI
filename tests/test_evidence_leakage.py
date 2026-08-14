import torch

from evidence_fixtures import evidence_config
from phase2_fixtures import synthetic_batch, synthetic_features
from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor


def test_evidence_extraction_is_target_independent_in_eval_mode() -> None:
    torch.manual_seed(5)
    extractor = HeterogeneousEvidenceExtractor(evidence_config(hidden_dim=7))
    extractor.eval()
    features = synthetic_features()
    clean_batch = synthetic_batch()
    changed_targets = synthetic_batch()
    changed_targets["image"] = clean_batch["image"].clone()
    changed_targets["binary_target"] = 1 - changed_targets["binary_target"]
    changed_targets["atomic_target"] = 1 - changed_targets["atomic_target"]
    changed_targets["fake_image_box_target"] = changed_targets["fake_image_box_target"] + 77.0
    changed_targets["manipulated_face_target_index"] = torch.tensor([2, 1, 0])
    changed_targets["fake_text_pos"] = torch.tensor([[4, 3], [2, 1], [1, 0]])

    with torch.no_grad():
        first = extractor(features, EvidenceStructuralMetadata.from_batch(clean_batch))
        second = extractor(features, EvidenceStructuralMetadata.from_batch(changed_targets))

    for family in first.families():
        first_set = first.families()[family]
        second_set = second.families()[family]
        assert torch.equal(first_set.mask, second_set.mask)
        assert torch.equal(first_set.source_indices, second_set.source_indices)
        assert torch.allclose(first_set.features, second_set.features)
        if first_set.word_ids is not None:
            assert torch.equal(first_set.word_ids, second_set.word_ids)
        if first_set.spatial_boxes is not None:
            assert torch.allclose(first_set.spatial_boxes, second_set.spatial_boxes)
    assert torch.allclose(
        first.cross_modal_metadata.similarity,
        second.cross_modal_metadata.similarity,
    )
