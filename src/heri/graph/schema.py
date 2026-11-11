"""Centralized graph schema."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias


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
SPATIAL_NEIGHBOR = "spatial_neighbor"
SEQUENTIAL_NEIGHBOR = "sequential_neighbor"
SUPPORTS_FS = "supports_fs"
SUPPORTS_FA = "supports_fa"
SUPPORTS_TS = "supports_ts"
SUPPORTS_TA = "supports_ta"
QUERIES_VISUAL_FORENSIC = "queries_visual_forensic"
QUERIES_TEXT_CONTEXT = "queries_text_context"
QUERIES_VISUAL_SEMANTIC = "queries_visual_semantic"
QUERIES_TEXT_ALIGNMENT = "queries_text_alignment"

_BASE_EDGE_TYPES: tuple[EdgeType, ...] = (
    (VISUAL_SEMANTIC, SPATIAL_CORRESPONDS, VISUAL_FORENSIC),
    (VISUAL_FORENSIC, SPATIAL_CORRESPONDS, VISUAL_SEMANTIC),
    (VISUAL_SEMANTIC, SEMANTIC_ALIGNS, TEXT_ALIGNMENT),
    (TEXT_ALIGNMENT, SEMANTIC_ALIGNS, VISUAL_SEMANTIC),
    (TEXT_ALIGNMENT, TOKEN_CONTEXTUALIZES, TEXT_CONTEXT),
    (TEXT_CONTEXT, TOKEN_CONTEXTUALIZES, TEXT_ALIGNMENT),
    (HYPOTHESIS, QUERIES_VISUAL_FORENSIC, VISUAL_FORENSIC),
    (HYPOTHESIS, QUERIES_TEXT_CONTEXT, TEXT_CONTEXT),
    (HYPOTHESIS, QUERIES_VISUAL_SEMANTIC, VISUAL_SEMANTIC),
    (HYPOTHESIS, QUERIES_TEXT_ALIGNMENT, TEXT_ALIGNMENT),
)
_INTRA_EDGE_TYPES: tuple[EdgeType, ...] = (
    (VISUAL_SEMANTIC, SPATIAL_NEIGHBOR, VISUAL_SEMANTIC),
    (VISUAL_FORENSIC, SPATIAL_NEIGHBOR, VISUAL_FORENSIC),
    (TEXT_ALIGNMENT, SEQUENTIAL_NEIGHBOR, TEXT_ALIGNMENT),
    (TEXT_CONTEXT, SEQUENTIAL_NEIGHBOR, TEXT_CONTEXT),
)


@dataclass(frozen=True)
class HypothesisSpec:
    """Dataset ontology consumed by the dataset-independent HERA core."""

    order: tuple[str, ...]
    image: tuple[str, ...]
    text: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "HypothesisSpec":
        mapping = value or {}
        order = tuple(str(name) for name in mapping.get("order", HYPOTHESIS_ORDER))
        image = tuple(str(name) for name in mapping.get("image", ("FS", "FA")))
        text = tuple(str(name) for name in mapping.get("text", ("TS", "TA")))
        if not order or len(set(order)) != len(order):
            raise ValueError("Hypothesis order must be non-empty and unique")
        unknown = (set(image) | set(text)).difference(order)
        if unknown:
            raise ValueError(f"Grounding hypothesis names are absent from order: {sorted(unknown)}")
        if not image or not text:
            raise ValueError("HERA requires at least one image and one text grounding hypothesis")
        return cls(order=order, image=image, text=text)

    @property
    def image_indices(self) -> tuple[int, ...]:
        return tuple(self.order.index(name) for name in self.image)

    @property
    def text_indices(self) -> tuple[int, ...]:
        return tuple(self.order.index(name) for name in self.text)


def support_relation_name(hypothesis: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", hypothesis.lower()).strip("_")
    if not token:
        raise ValueError(f"Invalid hypothesis name: {hypothesis!r}")
    return f"supports_{token}"


def edge_types_for_hypotheses(order: tuple[str, ...]) -> tuple[EdgeType, ...]:
    support_edges = tuple(
        (source, support_relation_name(name), HYPOTHESIS)
        for source in (VISUAL_FORENSIC, TEXT_CONTEXT, VISUAL_SEMANTIC, TEXT_ALIGNMENT)
        for name in order
    )
    # Preserve the historical DGM4 ordering for checkpoint/test compatibility.
    if order == HYPOTHESIS_ORDER:
        support_edges = (
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
        )
    return (*_BASE_EDGE_TYPES[:6], *_INTRA_EDGE_TYPES, *support_edges, *_BASE_EDGE_TYPES[6:])


CANONICAL_EDGE_TYPES: tuple[EdgeType, ...] = edge_types_for_hypotheses(HYPOTHESIS_ORDER)

SUPPORTED_EDGE_CONFIGS = {
    "visual_semantic_visual_forensic": {"spatial_knn"},
    "visual_semantic_text_alignment": {"similarity_topk"},
    "text_alignment_text_context": {"word_aligned", "span_aligned"},
}


def graph_metadata(edge_types: tuple[EdgeType, ...] = CANONICAL_EDGE_TYPES) -> tuple[list[NodeType], list[EdgeType]]:
    """Return metadata in the format expected by PyG HGTConv."""

    return list(NODE_TYPES), list(edge_types)


def validate_node_type(node_type: NodeType) -> None:
    """Fail fast for unknown node types."""

    if node_type not in NODE_TYPES:
        raise ValueError(f"Unknown graph node type: {node_type!r}")


def validate_edge_type(edge_type: EdgeType) -> None:
    """Fail fast for relation names outside the canonical schema."""

    if edge_type not in CANONICAL_EDGE_TYPES:
        raise ValueError(f"Unknown graph edge type: {edge_type!r}")


def support_relation_for_hypothesis(
    hypothesis_index: int,
    order: tuple[str, ...] = HYPOTHESIS_ORDER,
) -> RelationName:
    """Map a hypothesis index to its support relation name."""

    if hypothesis_index < 0 or hypothesis_index >= len(order):
        raise ValueError(f"Invalid hypothesis index: {hypothesis_index}")
    return support_relation_name(order[hypothesis_index])
