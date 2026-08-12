import pytest
import torch

from graph_fixtures import graph_config, synthetic_evidence
from src.graph.reasoner import HypothesisEvidenceGraphReasoner


def test_real_hgt_output_shape_when_backend_available() -> None:
    pytest.importorskip("torch_geometric")
    reasoner = HypothesisEvidenceGraphReasoner(graph_config(cycles=1))
    output = reasoner(synthetic_evidence())

    assert output.hypothesis_features.shape == (3, 4, 12)
    assert torch.isfinite(output.hypothesis_features).all()
    assert len(output.intermediate_hypothesis_features) == 1


def test_real_hgt_two_cycles_when_backend_available() -> None:
    pytest.importorskip("torch_geometric")
    reasoner = HypothesisEvidenceGraphReasoner(graph_config(cycles=2))
    output = reasoner(synthetic_evidence())

    assert output.hypothesis_features.shape == (3, 4, 12)
    assert len(output.intermediate_hypothesis_features) == 2
