from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from tests.support.graph import graph_config, synthetic_evidence
from heri.data.tokenizer_alignment import token_character_spans
from heri.evidence.budgeting import (
    LearnedEvidenceSelector,
    greedy_saliency_coverage_indices,
)
from heri.evidence.cross_modal import _native_global_calibration_loss
from heri.graph.builder import HypothesisEvidenceGraphBuilder
from heri.graph.edges import (
    span_aligned_pairs,
    intra_spatial_pairs,
    sequential_neighbor_pairs,
)
from heri.graph.hgt import EdgeAwareHGTGraphBlock, native_segment_softmax
from heri.graph.reasoner import HypothesisEvidenceGraphReasoner
from heri.graph.schema import HYPOTHESIS, VISUAL_SEMANTIC, TEXT_ALIGNMENT
from heri.routing import EvidenceReliabilityRouter
from tests.support.routing import routing_config


def test_eq30_router_regularization_matches_pooled_node_mean_and_detached_teacher():
    config = routing_config(hidden_dim=2)
    config["routing"]["regularization"] = {
        "enabled": True,
        "weight": 0.05,
        "target_mean": 0.75,
        "min_std": 0.05,
    }
    router = EvidenceReliabilityRouter(config)
    logits = torch.logit(torch.tensor([[0.25, 0.75]])).requires_grad_()
    selection = torch.tensor([[0.0, 1.0]], requires_grad=True)
    family = SimpleNamespace(
        original_features=torch.ones(1, 2, 2),
        mask=torch.ones(1, 2, dtype=torch.bool),
        reliability_logits=logits,
        reliability_scores=logits.sigmoid(),
        original=SimpleNamespace(selection_scores=selection),
    )
    actual = router._regularization_loss({"visual_semantic": family})
    expected = (-0.5 * math.log(0.25) - 0.5 * math.log(0.75) - math.log(0.75)) / 2 + 0.25**2
    assert actual.item() == pytest.approx(expected)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert selection.grad is None


def test_greedy_selection_trades_saliency_for_actual_feature_diversity_and_trains_gate():
    features = torch.tensor([[[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]]])
    scores = torch.tensor([0.9, 0.89, 0.8])
    selected = greedy_saliency_coverage_indices(torch.arange(3), scores, features[0], 2)
    assert selected.tolist() == [0, 2]
    selector = LearnedEvidenceSelector()
    output = selector(
        features,
        torch.ones(1, 3, dtype=torch.bool),
        max_nodes=2,
        strategy="greedy_saliency_coverage",
    )
    output.features.sum().backward()
    assert selector.scorer.weight.grad is not None
    assert torch.isfinite(selector.scorer.weight.grad).all()
    assert output.ungated_features is not None


def test_span_correspondence_uses_overlap_then_nearest_without_cross_sample_edges():
    pairs = span_aligned_pairs(
        src_spans=torch.tensor([[[0, 2], [2, 4], [9, 10]]]),
        src_mask=torch.ones(1, 3, dtype=torch.bool),
        dst_spans=torch.tensor([[[1, 3], [6, 8]]]),
        dst_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert list(zip(pairs.src_local.tolist(), pairs.dst_local.tolist())) == [
        (0, 0),
        (1, 0),
        (2, 1),
    ]
    assert token_character_spans(
        ["hello", "world"], [0, 0, None, 1], [(0, 2), (2, 5), (0, 0), (0, 3)]
    ) == [(0, 2), (2, 5), (-1, -1), (6, 9)]


def test_structural_neighbors_exclude_self_and_preserve_original_token_order():
    centers = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    pairs = intra_spatial_pairs(centers=centers, mask=mask, k=1)
    assert not bool((pairs.src_local == pairs.dst_local).any())
    edges = set(zip(pairs.src_local.tolist(), pairs.dst_local.tolist()))
    assert all((j, i) in edges for i, j in edges)
    sequence = sequential_neighbor_pairs(source_indices=torch.tensor([[8, 1, 4]]), mask=mask)
    assert set(zip(sequence.src_local.tolist(), sequence.dst_local.tolist())) == {
        (1, 2),
        (2, 1),
        (2, 0),
        (0, 2),
    }


def test_literal_alignment_denominator_and_empty_visual_store():
    similarity = torch.tensor([[[0.8, 0.6, 0.4], [0.1, 0.2, 0.3]]], requires_grad=True)
    valid = torch.ones_like(similarity, dtype=torch.bool)
    globals = torch.tensor([[1.0, 0.0]])
    loss = _native_global_calibration_loss(
        similarity=similarity,
        valid=valid,
        native_image_global=globals,
        native_text_global=globals,
    )
    assert torch.allclose(loss, torch.tensor(0.005), atol=1e-7)  # sum maxima / 2 = .9, target = 1
    empty = similarity[:, :0]
    zero = _native_global_calibration_loss(
        similarity=empty,
        valid=valid[:, :0],
        native_image_global=globals,
        native_text_global=globals,
    )
    assert zero.item() == 0 and zero.requires_grad


def test_native_segment_softmax_is_stable_differentiable_and_normalized_per_destination():
    logits = torch.tensor([[1000.0, -1000.0], [999.0, -999.0], [-4.0, 4.0]], requires_grad=True)
    index = torch.tensor([0, 0, 2])
    output = native_segment_softmax(logits, index, num_nodes=3)
    assert torch.allclose(output[:2], torch.softmax(logits[:2], dim=0))
    assert torch.equal(output[2], torch.ones(2))
    output.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_canonical_hgt_matches_confidence_log_prior_and_message_gate():
    edge = ("source", "supports", "destination")
    block = EdgeAwareHGTGraphBlock(
        hidden_dim=2,
        heads=1,
        dropout=0.0,
        edge_types=[edge],
        node_types=["source", "destination"],
    )
    with torch.no_grad():
        for bank in (block.query, block.key, block.value, block.output):
            for linear in bank.values():
                linear.weight.copy_(torch.eye(2))
                assert linear.bias is None
        block.relation_key[0].weight.copy_(torch.eye(2))
        block.relation_value[0].weight.copy_(torch.eye(2))
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    destination = torch.tensor([[0.3, -0.2]])
    confidence = torch.tensor([0.25, 0.8])
    actual = block(
        {"source": source, "destination": destination},
        {edge: torch.tensor([[0, 1], [0, 0]])},
        {edge: torch.stack([torch.zeros(2), confidence], dim=1)},
    )
    attention = torch.softmax(
        (source * destination).sum(dim=1) / (2**0.5) + torch.log(confidence + 1e-8),
        dim=0,
    )
    message = (source * (attention * confidence)[:, None]).sum(dim=0, keepdim=True)
    assert torch.allclose(
        actual["destination"],
        block.norms["destination"](destination + message),
        atol=1e-6,
    )
    assert not hasattr(block, "relation_priority") and not hasattr(block, "edge_attention_bias")


def test_graph_ontology_and_support_confidence_are_exact():
    evidence = synthetic_evidence()
    families = {}
    for name, value in evidence.families().items():
        families[name] = replace(value, reliability_scores=value.mask.float() * 0.7)
    evidence = replace(evidence, **families)
    cfg = graph_config()
    cfg["edges"]["feedback"]["semantic_feedback"] = True
    graph = HypothesisEvidenceGraphBuilder(cfg).build(evidence, torch.ones(3, 4, 12))
    assert graph.edge_index_dict[(VISUAL_SEMANTIC, "supports_ts", HYPOTHESIS)].numel() == 0
    assert graph.edge_index_dict[(TEXT_ALIGNMENT, "supports_fs", HYPOTHESIS)].numel() == 0
    relation = (VISUAL_SEMANTIC, "supports_fs", HYPOTHESIS)
    attrs = graph.edge_attr_dict[relation]
    assert torch.allclose(attrs[:, 1], 0.7 * ((1 + attrs[:, 0]) * 0.5), atol=1e-6)


@pytest.mark.parametrize("control", ["semantic", "hypothesis", "initial"])
def test_refresh_changes_only_authorized_relations_and_initial_control_uses_h0(control):
    torch.manual_seed(9)
    config = graph_config(cycles=2, dynamic_rebuild=True)
    config["reasoning"].update(
        refresh_semantic=control != "hypothesis",
        refresh_hypothesis=control != "semantic",
        hypothesis_state="initial" if control == "initial" else "evolved",
    )
    reasoner = HypothesisEvidenceGraphReasoner(config).eval()
    observed = []
    build = reasoner.builder.build

    def capture(*args, **kwargs):
        graph = build(*args, **kwargs)
        observed.append(graph)
        return graph

    reasoner.builder.build = capture
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12))
    output = reasoner(router(synthetic_evidence()).as_graph_evidence())
    first, final = observed[0], output.graph_batch
    semantic_relation = (VISUAL_SEMANTIC, "semantic_compares", TEXT_ALIGNMENT)
    support_relation = (VISUAL_SEMANTIC, "supports_fs", HYPOTHESIS)
    if control == "semantic":
        assert torch.equal(
            final.edge_attr_dict[support_relation],
            first.edge_attr_dict[support_relation],
        )
        assert not torch.allclose(
            final.edge_attr_dict[semantic_relation],
            first.edge_attr_dict[semantic_relation],
        )
    elif control == "hypothesis":
        assert torch.equal(
            final.edge_index_dict[semantic_relation],
            first.edge_index_dict[semantic_relation],
        )
        assert torch.equal(
            final.edge_attr_dict[semantic_relation],
            first.edge_attr_dict[semantic_relation],
        )
        assert not torch.allclose(
            final.edge_attr_dict[support_relation],
            first.edge_attr_dict[support_relation],
        )
    else:
        sources, destinations = final.edge_index_dict[support_relation]
        current_evidence = final.x_dict[VISUAL_SEMANTIC][sources]
        initial_hypotheses = first.x_dict[HYPOTHESIS][destinations]
        reference = torch.nn.functional.cosine_similarity(
            current_evidence.float(), initial_hypotheses.float(), dim=-1
        )
        torch.testing.assert_close(final.edge_attr_dict[support_relation][:, 0], reference)
        assert not torch.allclose(final.x_dict[HYPOTHESIS], first.x_dict[HYPOTHESIS])


@pytest.mark.parametrize(
    "mode",
    ["full", "static", "semantic", "hypothesis", "initial", "no_hypotheses", "pooled"],
)
def test_tiny_real_graph_controls_forward_and_backward(mode):
    cfg = graph_config(cycles=2, dynamic_rebuild=True)
    cfg["edges"]["text_alignment_text_context"]["method"] = "span_aligned"
    cfg["reasoning"].update(
        refresh_semantic=mode not in {"static", "hypothesis"},
        refresh_hypothesis=mode not in {"static", "semantic"},
        hypothesis_state="initial" if mode == "initial" else "evolved",
    )
    if mode == "static":
        cfg["reasoning"]["dynamic_rebuild"] = False
    if mode == "no_hypotheses":
        cfg["include_hypothesis_nodes"] = False
    if mode == "pooled":
        cfg["reasoning"].update(operator="pooled", cycles=1, dynamic_rebuild=False)
    reasoner = HypothesisEvidenceGraphReasoner(cfg)
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12))
    output = reasoner(router(synthetic_evidence()).as_graph_evidence())
    output.validate()
    output.hypothesis_features.square().mean().backward()
    assert output.hypothesis_features.shape == (3, 4, 12)
    assert all(torch.isfinite(p.grad).all() for p in reasoner.parameters() if p.grad is not None)
    assert output.graph_batch.x_dict[HYPOTHESIS].shape[0] == (
        0 if mode in {"no_hypotheses", "pooled"} else 12
    )
