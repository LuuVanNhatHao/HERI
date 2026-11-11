from dataclasses import replace

import torch

from tests.support.graph import graph_config, synthetic_evidence
from heri.evidence.outputs import CrossModalMetadata
from heri.graph.builder import HypothesisEvidenceGraphBuilder
from heri.graph.hypotheses import HypothesisNodeInitializer
from heri.graph.schema import (
    HYPOTHESIS,
    QUERIES_TEXT_CONTEXT,
    QUERIES_VISUAL_FORENSIC,
    SEMANTIC_ALIGNS,
    SPATIAL_CORRESPONDS,
    SUPPORTS_FA,
    SUPPORTS_FS,
    SUPPORTS_TA,
    SUPPORTS_TS,
    TEXT_ALIGNMENT,
    TEXT_CONTEXT,
    TOKEN_CONTEXTUALIZES,
    VISUAL_FORENSIC,
    VISUAL_SEMANTIC,
)


def build_graph():
    config = graph_config()
    evidence = synthetic_evidence()
    hypotheses = HypothesisNodeInitializer(config)(batch_size=3)
    return HypothesisEvidenceGraphBuilder(config).build(evidence, hypotheses)


def test_graph_builder_node_counts_and_core_edges() -> None:
    graph = build_graph()

    graph.validate()
    assert graph.x_dict[VISUAL_SEMANTIC].shape == (9, 12)
    assert graph.x_dict[VISUAL_FORENSIC].shape == (9, 12)
    assert "face" not in graph.x_dict
    assert graph.x_dict[TEXT_ALIGNMENT].shape == (9, 12)
    assert graph.x_dict[TEXT_CONTEXT].shape == (9, 12)
    assert graph.x_dict[HYPOTHESIS].shape == (12, 12)
    assert (
        graph.edge_index_dict[(VISUAL_SEMANTIC, SPATIAL_CORRESPONDS, VISUAL_FORENSIC)].shape[1]
        == 18
    )
    assert graph.edge_index_dict[(VISUAL_SEMANTIC, SEMANTIC_ALIGNS, TEXT_ALIGNMENT)].shape[1] == 18
    assert graph.edge_index_dict[(TEXT_ALIGNMENT, TOKEN_CONTEXTUALIZES, TEXT_CONTEXT)].shape[1] == 6
    semantic_attrs = graph.edge_attr_dict[(VISUAL_SEMANTIC, SEMANTIC_ALIGNS, TEXT_ALIGNMENT)]
    assert semantic_attrs.shape == (18, 2)
    assert torch.any(semantic_attrs[:, 0] != 0)
    assert torch.allclose(
        semantic_attrs[:, 1], ((1.0 + semantic_attrs[:, 0]) * 0.5).clamp_min(1e-8)
    )
    assert SEMANTIC_ALIGNS == "semantic_compares"

    token_edges = graph.edge_index_dict[(TEXT_ALIGNMENT, TOKEN_CONTEXTUALIZES, TEXT_CONTEXT)]
    src_local = graph.node_local_indices[TEXT_ALIGNMENT][token_edges[0]]
    dst_local = graph.node_local_indices[TEXT_CONTEXT][token_edges[1]]
    src_samples = graph.node_sample_indices[TEXT_ALIGNMENT][token_edges[0]]
    evidence = synthetic_evidence()
    assert torch.equal(
        evidence.text_alignment.word_ids[src_samples, src_local],
        evidence.text_context.word_ids[src_samples, dst_local],
    )


def test_hypothesis_support_and_feedback_relations() -> None:
    graph = build_graph()

    assert graph.edge_index_dict[(VISUAL_FORENSIC, SUPPORTS_FS, HYPOTHESIS)].shape[1] == 9
    assert graph.edge_index_dict[(VISUAL_FORENSIC, SUPPORTS_FA, HYPOTHESIS)].shape[1] == 9
    assert graph.edge_index_dict[(TEXT_CONTEXT, SUPPORTS_TS, HYPOTHESIS)].shape[1] == 9
    assert graph.edge_index_dict[(TEXT_CONTEXT, SUPPORTS_TA, HYPOTHESIS)].shape[1] == 9
    assert graph.edge_index_dict[(VISUAL_SEMANTIC, SUPPORTS_FS, HYPOTHESIS)].shape[1] == 9
    assert graph.edge_index_dict[(TEXT_ALIGNMENT, SUPPORTS_TA, HYPOTHESIS)].shape[1] == 9
    assert (
        graph.edge_index_dict[(HYPOTHESIS, QUERIES_VISUAL_FORENSIC, VISUAL_FORENSIC)].shape[1] == 18
    )
    assert graph.edge_index_dict[(HYPOTHESIS, QUERIES_TEXT_CONTEXT, TEXT_CONTEXT)].shape[1] == 18


def test_masked_similarity_zero_does_not_create_edges() -> None:
    config = graph_config()
    evidence = synthetic_evidence()
    text_mask = evidence.text_alignment.mask.clone()
    text_mask[0, 2] = False
    text_sources = evidence.text_alignment.source_indices.clone()
    text_sources[0, 2] = -1
    text_alignment = replace(
        evidence.text_alignment,
        mask=text_mask,
        source_indices=text_sources,
    )
    cross = CrossModalMetadata(
        similarity=torch.zeros_like(evidence.cross_modal_metadata.similarity),
        visual_mask=evidence.visual_semantic.mask,
        text_mask=text_mask,
        visual_source_indices=evidence.visual_semantic.source_indices,
        text_source_indices=text_sources,
        method="cosine",
    )
    evidence = replace(evidence, text_alignment=text_alignment, cross_modal_metadata=cross)
    graph = HypothesisEvidenceGraphBuilder(config).build(
        evidence,
        HypothesisNodeInitializer(config)(batch_size=3),
    )

    edge_index = graph.edge_index_dict[(VISUAL_SEMANTIC, SEMANTIC_ALIGNS, TEXT_ALIGNMENT)]
    dst_samples = graph.node_sample_indices[TEXT_ALIGNMENT][edge_index[1]]
    dst_locals = graph.node_local_indices[TEXT_ALIGNMENT][edge_index[1]]
    assert not bool(((dst_samples == 0) & (dst_locals == 2)).any())
