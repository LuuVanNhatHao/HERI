import pytest
import torch

from graph_fixtures import graph_config, synthetic_evidence
from src.graph.hgt import _accumulate_edge_messages
from src.graph.reasoner import HypothesisEvidenceGraphReasoner


def test_edge_message_accumulation_casts_bf16_autocast_output_to_fp32_residual() -> None:
    accumulator = torch.zeros(3, 2, dtype=torch.float32)
    counts = torch.zeros(3, 1, dtype=torch.float32)
    destinations = torch.tensor([0, 2, 2])
    messages = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16)

    _accumulate_edge_messages(accumulator, counts, destinations, messages)

    assert accumulator.dtype == torch.float32
    assert counts.dtype == torch.float32
    assert torch.equal(accumulator, torch.tensor([[1, 2], [0, 0], [8, 10]], dtype=torch.float32))
    assert torch.equal(counts, torch.tensor([[1], [0], [2]], dtype=torch.float32))


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
