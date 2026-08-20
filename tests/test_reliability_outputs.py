from __future__ import annotations

import torch

from graph_fixtures import synthetic_evidence
from routing_fixtures import routing_config
from src.routing import EvidenceReliabilityRouter


def test_routed_output_preserves_all_families_and_shapes() -> None:
    evidence = synthetic_evidence(hidden_dim=12)
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(evidence)
    routed.validate()
    assert set(routed.families()) == {"visual_semantic", "visual_forensic", "text_alignment", "text_context"}
    for name, routed_set in routed.families().items():
        original = evidence.families()[name]
        assert routed_set.original_features.shape == original.features.shape
        assert routed_set.routed_features.shape == original.features.shape
        assert routed_set.reliability_logits.shape == original.mask.shape
        assert routed_set.reliability_scores.shape == original.mask.shape
        assert routed_set.provenance == original.provenance


def test_reliability_scores_are_masked_probabilities() -> None:
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(synthetic_evidence(hidden_dim=12))
    for routed_set in routed.families().values():
        assert torch.all((routed_set.reliability_scores >= 0) & (routed_set.reliability_scores <= 1))
        assert torch.all(routed_set.reliability_scores[~routed_set.mask] == 0)
        assert torch.all(routed_set.routed_features[~routed_set.mask] == 0)


def test_as_graph_evidence_is_phase4_compatible() -> None:
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(synthetic_evidence(hidden_dim=12))
    graph_evidence = routed.as_graph_evidence()
    graph_evidence.validate()
    assert graph_evidence.text_context.full_features is routed.original.text_context.full_features
