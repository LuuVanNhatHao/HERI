from __future__ import annotations

import torch

from decision_fixtures import decision_config, graph_output_from_evidence
from graph_fixtures import synthetic_evidence
from routing_fixtures import routing_config
from src.models.hera import HERADecisionModule
from src.routing import EvidenceReliabilityRouter


def test_router_bypass_preserves_valid_features_exactly() -> None:
    evidence = synthetic_evidence(hidden_dim=12)
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12, enabled=False))(evidence)
    for name, routed_set in routed.families().items():
        original = evidence.families()[name]
        assert torch.equal(routed_set.reliability_scores[original.mask], torch.ones_like(routed_set.reliability_scores[original.mask]))
        assert torch.equal(routed_set.routed_features[original.mask], original.features[original.mask])
    routed.as_graph_evidence().validate()


def test_router_bypass_equivalence_to_non_routed_decision_path() -> None:
    evidence = synthetic_evidence(hidden_dim=12)
    routed_evidence = EvidenceReliabilityRouter(routing_config(hidden_dim=12, enabled=False))(evidence).as_graph_evidence()
    original_graph = graph_output_from_evidence(evidence, hidden_dim=12)
    routed_graph = graph_output_from_evidence(routed_evidence, hidden_dim=12)
    module = HERADecisionModule(decision_config()).eval()
    original_output = module(original_graph)
    routed_output = module(routed_graph)
    assert torch.allclose(original_output.atomic_logits, routed_output.atomic_logits)
    assert torch.allclose(original_output.binary_probability, routed_output.binary_probability)
    assert torch.allclose(original_output.image_grounding_logits, routed_output.image_grounding_logits)
    assert torch.allclose(original_output.text_grounding_logits, routed_output.text_grounding_logits)
