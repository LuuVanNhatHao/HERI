import torch

from graph_fixtures import graph_config, synthetic_evidence
from phase2_fixtures import synthetic_batch
from src.graph.builder import HypothesisEvidenceGraphBuilder
from src.graph.hypotheses import HypothesisNodeInitializer


def test_graph_construction_is_target_independent() -> None:
    config = graph_config()
    evidence = synthetic_evidence()
    hypotheses = HypothesisNodeInitializer(config)(batch_size=3)
    changed_targets = synthetic_batch()
    changed_targets["binary_target"] = 1 - changed_targets["binary_target"]
    changed_targets["atomic_target"] = 1 - changed_targets["atomic_target"]
    changed_targets["fake_cls"] = ["text_attribute", "orig", "face_swap"]
    changed_targets["fake_image_box_target"] = changed_targets["fake_image_box_target"] + 123.0
    changed_targets["manipulated_face_target_index"] = torch.tensor([2, 1, 0])
    changed_targets["fake_text_pos"] = torch.tensor([[3, 2], [1, 0], [4, 4]])

    builder = HypothesisEvidenceGraphBuilder(config)
    first = builder.build(evidence, hypotheses)
    second = builder.build(evidence, hypotheses)

    for node_type in first.x_dict:
        assert torch.equal(first.x_dict[node_type], second.x_dict[node_type])
        assert torch.equal(first.node_sample_indices[node_type], second.node_sample_indices[node_type])
        assert torch.equal(first.node_local_indices[node_type], second.node_local_indices[node_type])
    for edge_type in first.edge_index_dict:
        assert torch.equal(first.edge_index_dict[edge_type], second.edge_index_dict[edge_type])
        assert torch.equal(first.edge_sample_indices[edge_type], second.edge_sample_indices[edge_type])


def test_cross_sample_isolation_and_sample_a_stability() -> None:
    config = graph_config()
    evidence = synthetic_evidence()
    hypotheses = HypothesisNodeInitializer(config)(batch_size=3)
    graph = HypothesisEvidenceGraphBuilder(config).build(evidence, hypotheses)

    for edge_type, edge_index in graph.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        src_samples = graph.node_sample_indices[src_type][edge_index[0]]
        dst_samples = graph.node_sample_indices[dst_type][edge_index[1]]
        assert torch.equal(src_samples, dst_samples)

    changed_evidence = synthetic_evidence()
    changed_evidence.visual_semantic.features[1:] = changed_evidence.visual_semantic.features[1:] * -5.0
    changed_graph = HypothesisEvidenceGraphBuilder(config).build(changed_evidence, hypotheses)
    for node_type in graph.x_dict:
        sample_zero = graph.node_sample_indices[node_type] == 0
        changed_sample_zero = changed_graph.node_sample_indices[node_type] == 0
        assert torch.allclose(graph.x_dict[node_type][sample_zero], changed_graph.x_dict[node_type][changed_sample_zero])


def test_zero_face_family_does_not_fabricate_nodes() -> None:
    config = graph_config()
    evidence = synthetic_evidence()
    assert evidence.face.features.shape[1] == 0
    graph = HypothesisEvidenceGraphBuilder(config).build(
        evidence,
        HypothesisNodeInitializer(config)(batch_size=3),
    )

    assert graph.x_dict["face"].shape == (0, 12)
    assert graph.x_dict["hypothesis"].shape == (12, 12)
    graph.validate()
