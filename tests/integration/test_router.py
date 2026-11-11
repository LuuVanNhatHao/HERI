from __future__ import annotations

import torch

from tests.support.decision import decision_config, graph_output_from_evidence
from tests.support.graph import graph_config, synthetic_evidence
from tests.support.routing import routing_config
from heri.graph.builder import HypothesisEvidenceGraphBuilder
from heri.graph.schema import HYPOTHESIS_ORDER
from heri.models.heads.noisy_or import noisy_or_from_probabilities
from heri.models.heri import HERADecisionModule
from heri.routing import EvidenceReliabilityRouter


def test_routed_evidence_builds_typed_graph() -> None:
    routed_evidence = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(
        synthetic_evidence(hidden_dim=12)
    ).as_graph_evidence()
    hypotheses = torch.randn(3, len(HYPOTHESIS_ORDER), 12)
    graph = HypothesisEvidenceGraphBuilder(graph_config(hidden_dim=12)).build(
        routed_evidence, hypotheses
    )
    graph.validate()


def test_routed_graph_output_is_decision_compatible_and_noisy_or_unchanged() -> None:
    routed_evidence = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(
        synthetic_evidence(hidden_dim=12)
    ).as_graph_evidence()
    graph_output = graph_output_from_evidence(routed_evidence, hidden_dim=12)
    output = HERADecisionModule(decision_config()).eval()(graph_output)
    expected = noisy_or_from_probabilities(output.atomic_probabilities)
    assert torch.allclose(output.binary_probability, expected, atol=1e-6)
