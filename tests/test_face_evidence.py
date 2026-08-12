import torch

from evidence_fixtures import evidence_config
from phase2_fixtures import synthetic_batch, synthetic_features
from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor


def test_face_evidence_uses_candidate_indices_and_tiny_face_fallback() -> None:
    extractor = HeterogeneousEvidenceExtractor(evidence_config(hidden_dim=10))
    batch = synthetic_batch()
    features = synthetic_features()
    features.dinov3.patch_tokens = torch.tensor(
        [
            [[1.0] * 17, [2.0] * 17, [3.0] * 17, [4.0] * 17],
            [[5.0] * 17, [6.0] * 17, [7.0] * 17, [8.0] * 17],
            [[9.0] * 17, [10.0] * 17, [11.0] * 17, [12.0] * 17],
        ]
    )

    evidence = extractor(features, EvidenceStructuralMetadata.from_batch(batch))

    assert evidence.face.provenance["tiny_face_fallback"] == "nearest_patch_center"
    assert evidence.face.source_indices.tolist()[1] == [0, -1, -1]
    assert evidence.face.normalized_boxes.shape == (3, 3, 4)
    valid_boxes = evidence.face.normalized_boxes[evidence.face.mask]
    assert torch.all((valid_boxes >= 0) & (valid_boxes <= 1))
