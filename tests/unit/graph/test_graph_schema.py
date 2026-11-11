import pytest
import torch

from tests.support.graph import graph_config
from heri.graph.builder import HypothesisEvidenceGraphBuilder
from heri.graph.hypotheses import HypothesisNodeInitializer
from heri.graph.schema import (
    CANONICAL_EDGE_TYPES,
    HYPOTHESIS_ORDER,
    NODE_TYPES,
    validate_edge_type,
)


def test_graph_schema_and_hypothesis_order_are_frozen() -> None:
    assert NODE_TYPES == (
        "visual_semantic",
        "visual_forensic",
        "text_alignment",
        "text_context",
        "hypothesis",
    )
    assert HYPOTHESIS_ORDER == ("FS", "FA", "TS", "TA")
    assert all("face" not in edge_type for edge_type in CANONICAL_EDGE_TYPES)
    assert ("text_context", "supports_ta", "hypothesis") in CANONICAL_EDGE_TYPES


def test_unknown_relation_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown graph edge type"):
        validate_edge_type(("unknown", "edge_1", "hypothesis"))


def test_unknown_edge_config_key_fails_fast() -> None:
    config = graph_config()
    config["edges"]["edge_1"] = {"method": "full_bipartite"}

    with pytest.raises(ValueError, match="Unknown graph edge config keys"):
        HypothesisEvidenceGraphBuilder(config)


def test_learned_hypothesis_initialization_is_label_agnostic() -> None:
    torch.manual_seed(3)
    initializer = HypothesisNodeInitializer(graph_config())
    features = initializer(batch_size=2)

    assert features.shape == (2, 4, 12)
    assert torch.allclose(features[0], features[1])
    assert initializer.embeddings.requires_grad
