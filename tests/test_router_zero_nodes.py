from __future__ import annotations

import torch

from decision_fixtures import zero_face_graph_output
from graph_fixtures import synthetic_evidence
from routing_fixtures import routing_config
from src.routing import EvidenceReliabilityRouter


def test_zero_face_family_is_safe() -> None:
    graph_output = zero_face_graph_output()
    evidence = synthetic_evidence(hidden_dim=12)
    evidence = type(evidence)(
        visual_semantic=evidence.visual_semantic,
        visual_forensic=evidence.visual_forensic,
        face=graph_output.refined_face,
        text_alignment=evidence.text_alignment,
        text_context=evidence.text_context,
        cross_modal_metadata=evidence.cross_modal_metadata,
    )
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(evidence)
    assert routed.face.reliability_scores.shape == (3, 0)
    assert routed.face.routed_features.shape == (3, 0, 12)
    assert torch.isfinite(routed.face.routed_features).all()
    routed.as_graph_evidence().validate()
