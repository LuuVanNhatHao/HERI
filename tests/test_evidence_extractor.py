import torch

from evidence_fixtures import evidence_config
from phase2_fixtures import synthetic_batch, synthetic_features
from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor


def test_full_heterogeneous_evidence_contract() -> None:
    extractor = HeterogeneousEvidenceExtractor(evidence_config())
    batch = synthetic_batch()
    features = synthetic_features()

    evidence = extractor(features, EvidenceStructuralMetadata.from_batch(batch))

    evidence.validate()
    assert evidence.visual_semantic.features.shape == (3, 3, 12)
    assert evidence.visual_forensic.features.shape == (3, 3, 12)
    assert evidence.face.features.shape == (3, 3, 12)
    assert evidence.text_alignment.features.shape == (3, 3, 12)
    assert evidence.text_context.features.shape == (3, 3, 12)
    assert evidence.visual_semantic.source_indices.tolist()[0] == [0, 2, 3]
    assert evidence.visual_forensic.source_indices.tolist()[0] == [0, 2, 3]
    assert evidence.face.source_indices.tolist()[1] == [0, -1, -1]
    assert evidence.text_context.word_ids.tolist()[2] == [0, 2, 3]
    assert evidence.text_context.full_features is features.deberta.token_features
    assert evidence.cross_modal_metadata.similarity.shape == (3, 3, 3)
    invalid = ~(
        evidence.cross_modal_metadata.visual_mask.unsqueeze(2)
        & evidence.cross_modal_metadata.text_mask.unsqueeze(1)
    )
    assert torch.all(evidence.cross_modal_metadata.similarity[invalid] == 0)


def test_extractor_handles_no_valid_faces() -> None:
    extractor = HeterogeneousEvidenceExtractor(evidence_config())
    batch = synthetic_batch()
    batch["candidate_face_mask"] = torch.zeros_like(batch["candidate_face_mask"])
    batch["candidate_face_boxes"] = torch.zeros_like(batch["candidate_face_boxes"])

    evidence = extractor(synthetic_features(), EvidenceStructuralMetadata.from_batch(batch))

    assert evidence.face.features.shape == (3, 0, 12)
    assert evidence.face.mask.shape == (3, 0)
    evidence.validate()
