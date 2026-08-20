"""Centralized Phase-4 graph schema."""

from __future__ import annotations

from typing import TypeAlias


NodeType: TypeAlias = str
RelationName: TypeAlias = str
EdgeType: TypeAlias = tuple[NodeType, RelationName, NodeType]


VISUAL_SEMANTIC = "visual_semantic"
VISUAL_FORENSIC = "visual_forensic"
TEXT_ALIGNMENT = "text_alignment"
TEXT_CONTEXT = "text_context"
HYPOTHESIS = "hypothesis"

EVIDENCE_NODE_TYPES: tuple[NodeType, ...] = (
    VISUAL_SEMANTIC,
    VISUAL_FORENSIC,
    TEXT_ALIGNMENT,
    TEXT_CONTEXT,
)
NODE_TYPES: tuple[NodeType, ...] = (*EVIDENCE_NODE_TYPES, HYPOTHESIS)

HYPOTHESIS_ORDER: tuple[str, ...] = ("FS", "FA", "TS", "TA")
HYPOTHESIS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(HYPOTHESIS_ORDER)}

SPATIAL_CORRESPONDS = "spatial_corresponds"
SEMANTIC_ALIGNS = "semantic_compares"
TOKEN_CONTEXTUALIZES = "token_contextualizes"
SUPPORTS_FS = "supports_fs"
SUPPORTS_FA = "supports_fa"
SUPPORTS_TS = "supports_ts"
SUPPORTS_TA = "supports_ta"
QUERIES_VISUAL_FORENSIC = "queries_visual_forensic"
QUERIES_TEXT_CONTEXT = "queries_text_context"
QUERIES_VISUAL_SEMANTIC = "queries_visual_semantic"
QUERIES_TEXT_ALIGNMENT = "queries_text_alignment"

CANONICAL_EDGE_TYPES: tuple[EdgeType, ...] = (
    (VISUAL_SEMANTIC, SPATIAL_CORRESPONDS, VISUAL_FORENSIC),
    (VISUAL_FORENSIC, SPATIAL_CORRESPONDS, VISUAL_SEMANTIC),
    (VISUAL_SEMANTIC, SEMANTIC_ALIGNS, TEXT_ALIGNMENT),
    (TEXT_ALIGNMENT, SEMANTIC_ALIGNS, VISUAL_SEMANTIC),
    (TEXT_ALIGNMENT, TOKEN_CONTEXTUALIZES, TEXT_CONTEXT),
    (TEXT_CONTEXT, TOKEN_CONTEXTUALIZES, TEXT_ALIGNMENT),
    (VISUAL_FORENSIC, SUPPORTS_FS, HYPOTHESIS),
    (VISUAL_FORENSIC, SUPPORTS_FA, HYPOTHESIS),
    (TEXT_CONTEXT, SUPPORTS_TS, HYPOTHESIS),
    (TEXT_CONTEXT, SUPPORTS_TA, HYPOTHESIS),
    (VISUAL_SEMANTIC, SUPPORTS_FS, HYPOTHESIS),
    (VISUAL_SEMANTIC, SUPPORTS_FA, HYPOTHESIS),
    (VISUAL_SEMANTIC, SUPPORTS_TS, HYPOTHESIS),
    (VISUAL_SEMANTIC, SUPPORTS_TA, HYPOTHESIS),
    (TEXT_ALIGNMENT, SUPPORTS_FS, HYPOTHESIS),
    (TEXT_ALIGNMENT, SUPPORTS_FA, HYPOTHESIS),
    (TEXT_ALIGNMENT, SUPPORTS_TS, HYPOTHESIS),
    (TEXT_ALIGNMENT, SUPPORTS_TA, HYPOTHESIS),
    (HYPOTHESIS, QUERIES_VISUAL_FORENSIC, VISUAL_FORENSIC),
    (HYPOTHESIS, QUERIES_TEXT_CONTEXT, TEXT_CONTEXT),
    (HYPOTHESIS, QUERIES_VISUAL_SEMANTIC, VISUAL_SEMANTIC),
    (HYPOTHESIS, QUERIES_TEXT_ALIGNMENT, TEXT_ALIGNMENT),
)

SUPPORTED_EDGE_CONFIGS = {
    "visual_semantic_visual_forensic": {"spatial_knn"},
    "visual_semantic_text_alignment": {"similarity_topk"},
    "text_alignment_text_context": {"word_aligned"},
}


def graph_metadata() -> tuple[list[NodeType], list[EdgeType]]:
    """Return metadata in the format expected by PyG HGTConv."""

    return list(NODE_TYPES), list(CANONICAL_EDGE_TYPES)


def validate_node_type(node_type: NodeType) -> None:
    """Fail fast for unknown node types."""

    if node_type not in NODE_TYPES:
        raise ValueError(f"Unknown graph node type: {node_type!r}")


def validate_edge_type(edge_type: EdgeType) -> None:
    """Fail fast for relation names outside the canonical schema."""

    if edge_type not in CANONICAL_EDGE_TYPES:
        raise ValueError(f"Unknown graph edge type: {edge_type!r}")


def support_relation_for_hypothesis(hypothesis_index: int) -> RelationName:
    """Map a hypothesis index to its support relation name."""

    mapping = {
        HYPOTHESIS_TO_INDEX["FS"]: SUPPORTS_FS,
        HYPOTHESIS_TO_INDEX["FA"]: SUPPORTS_FA,
        HYPOTHESIS_TO_INDEX["TS"]: SUPPORTS_TS,
        HYPOTHESIS_TO_INDEX["TA"]: SUPPORTS_TA,
    }
    if hypothesis_index not in mapping:
        raise ValueError(f"Invalid hypothesis index: {hypothesis_index}")
    return mapping[hypothesis_index]
