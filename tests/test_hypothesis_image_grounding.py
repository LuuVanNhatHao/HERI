from __future__ import annotations

import torch

from decision_fixtures import synthetic_graph_output, zero_face_graph_output
from src.models.heads.hypothesis_grounding import HypothesisConditionedImageGroundingHead


def test_image_grounding_scores_fs_and_fa_faces() -> None:
    graph_output = synthetic_graph_output()
    head = HypothesisConditionedImageGroundingHead(hidden_dim=12, dropout=0.0)
    logits, probabilities, mask = head(
        hypothesis_features=graph_output.hypothesis_features,
        face_evidence=graph_output.refined_face,
    )
    assert logits.shape == (3, 2, 3)
    assert probabilities.shape == logits.shape
    assert mask.tolist() == [[True, True, False], [True, False, False], [True, True, True]]
    assert torch.all(probabilities.masked_select(~mask.unsqueeze(1)) == 0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3, 2), atol=1e-6)


def test_image_grounding_handles_zero_face_candidates() -> None:
    graph_output = zero_face_graph_output()
    head = HypothesisConditionedImageGroundingHead(hidden_dim=12, dropout=0.0)
    logits, probabilities, mask = head(
        hypothesis_features=graph_output.hypothesis_features,
        face_evidence=graph_output.refined_face,
    )
    assert logits.shape == (3, 2, 0)
    assert probabilities.shape == (3, 2, 0)
    assert mask.shape == (3, 0)


def test_image_grounding_uses_corresponding_hypothesis_state() -> None:
    graph_output = synthetic_graph_output()
    head = HypothesisConditionedImageGroundingHead(hidden_dim=12, dropout=0.0).eval()
    base_logits, _, _ = head(
        hypothesis_features=graph_output.hypothesis_features,
        face_evidence=graph_output.refined_face,
    )
    altered = graph_output.hypothesis_features.clone()
    altered[:, 0, :] += torch.randn_like(altered[:, 0, :]) * 3.0
    changed_logits, _, _ = head(hypothesis_features=altered, face_evidence=graph_output.refined_face)
    assert not torch.allclose(base_logits[:, 0, :], changed_logits[:, 0, :])
    assert torch.allclose(base_logits[:, 1, :], changed_logits[:, 1, :])
