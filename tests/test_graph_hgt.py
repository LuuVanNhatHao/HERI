import pytest
import torch

from graph_fixtures import graph_config, synthetic_evidence
from src.graph.hgt import EdgeAwareHGTGraphBlock, _accumulate_edge_messages
from src.graph.reasoner import HypothesisEvidenceGraphReasoner
from src.graph.schema import CANONICAL_EDGE_TYPES, HYPOTHESIS, NODE_TYPES, SUPPORTS_FS, VISUAL_SEMANTIC


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


def test_edge_confidence_changes_hgt_attention_and_message() -> None:
    pytest.importorskip("torch_geometric")
    torch.manual_seed(211)
    block = EdgeAwareHGTGraphBlock(hidden_dim=12, heads=3, dropout=0.0).eval()
    x_dict = {node: torch.empty(0, 12) for node in NODE_TYPES}
    x_dict[VISUAL_SEMANTIC] = torch.randn(2, 12)
    x_dict[HYPOTHESIS] = torch.randn(1, 12)
    relation = (VISUAL_SEMANTIC, SUPPORTS_FS, HYPOTHESIS)
    edge_indices = {
        edge: torch.empty(2, 0, dtype=torch.long) for edge in CANONICAL_EDGE_TYPES
    }
    edge_indices[relation] = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    full_confidence = {
        edge: torch.empty(0, 2) for edge in CANONICAL_EDGE_TYPES
    }
    full_confidence[relation] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    gated_confidence = {edge: value.clone() for edge, value in full_confidence.items()}
    gated_confidence[relation][1, 1] = 0.0

    full = block(x_dict, edge_indices, full_confidence)[HYPOTHESIS]
    gated = block(x_dict, edge_indices, gated_confidence)[HYPOTHESIS]

    assert not torch.allclose(full, gated)


def test_dynamic_two_cycle_reasoning_rebuilds_graph_from_refined_state() -> None:
    pytest.importorskip("torch_geometric")
    reasoner = HypothesisEvidenceGraphReasoner(
        graph_config(cycles=2, dynamic_rebuild=True)
    )
    built_graphs = []
    original_build = reasoner.builder.build

    def capture_build(evidence, hypotheses):
        graph = original_build(evidence, hypotheses)
        built_graphs.append(graph)
        return graph

    reasoner.builder.build = capture_build
    output = reasoner(synthetic_evidence())

    assert len(built_graphs) == 2
    assert output.graph_statistics["dynamic_rebuild"] is True
    assert len(output.graph_statistics["cycle_statistics"]) == 2
    relation = (VISUAL_SEMANTIC, SUPPORTS_FS, HYPOTHESIS)
    assert not torch.allclose(
        built_graphs[0].edge_attr_dict[relation],
        built_graphs[1].edge_attr_dict[relation],
    )
