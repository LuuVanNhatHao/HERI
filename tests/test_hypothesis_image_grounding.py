from __future__ import annotations

import torch

from decision_fixtures import synthetic_graph_output
from src.models.heads.hypothesis_grounding import HypothesisConditionedImageGroundingHead


def test_image_grounding_scores_dense_dino_patches_and_predicts_boxes() -> None:
    graph_output = synthetic_graph_output()
    head = HypothesisConditionedImageGroundingHead(hidden_dim=12, dropout=0.0)
    logits, probabilities, mask, boxes = head(
        hypothesis_features=graph_output.hypothesis_features,
        image_evidence=graph_output.refined_visual_forensic,
    )
    assert logits.shape == (3, 2, 4)
    assert probabilities.shape == logits.shape
    assert mask.tolist() == [[True] * 4] * 3
    assert boxes.shape == (3, 2, 4)
    assert torch.all((boxes >= 0) & (boxes <= 1))
    assert torch.all(boxes[..., 2:] >= boxes[..., :2])
    assert torch.all(probabilities.masked_select(~mask.unsqueeze(1)) == 0)


def test_image_grounding_requires_no_annotation_face_candidates() -> None:
    graph_output = synthetic_graph_output()
    head = HypothesisConditionedImageGroundingHead(hidden_dim=12, dropout=0.0)
    logits, probabilities, mask, boxes = head(
        hypothesis_features=graph_output.hypothesis_features,
        image_evidence=graph_output.refined_visual_forensic,
    )
    assert logits.shape[-1] == 4
    assert probabilities.shape == logits.shape
    assert mask.shape == (3, 4)
    assert boxes.shape == (3, 2, 4)


def test_image_grounding_uses_corresponding_hypothesis_state() -> None:
    graph_output = synthetic_graph_output()
    head = HypothesisConditionedImageGroundingHead(hidden_dim=12, dropout=0.0).eval()
    base_logits, _, _, _ = head(
        hypothesis_features=graph_output.hypothesis_features,
        image_evidence=graph_output.refined_visual_forensic,
    )
    altered = graph_output.hypothesis_features.clone()
    altered[:, 0, :] += torch.randn_like(altered[:, 0, :]) * 3.0
    changed_logits, _, _, _ = head(
        hypothesis_features=altered,
        image_evidence=graph_output.refined_visual_forensic,
    )
    assert not torch.allclose(base_logits[:, 0, :], changed_logits[:, 0, :])
    assert torch.allclose(base_logits[:, 1, :], changed_logits[:, 1, :])
