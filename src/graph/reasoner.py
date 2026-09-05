"""Phase-4 hypothesis-evidence graph reasoner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from torch import nn

from src.evidence.outputs import CrossModalMetadata, EvidenceSet, HeterogeneousEvidenceBatch

from .adapters import TypedEvidenceFeatureAdapters
from .ablations import GATv2GraphEncoder, HomogeneousEvidenceHGTEncoder
from .builder import HypothesisEvidenceGraphBuilder
from .hgt import HGTGraphEncoder
from .hypotheses import HypothesisNodeInitializer
from .outputs import HypothesisGraphOutput
from .schema import HYPOTHESIS, HYPOTHESIS_ORDER
from .schema import TEXT_ALIGNMENT, TEXT_CONTEXT, VISUAL_FORENSIC, VISUAL_SEMANTIC
from .validation import graph_statistics


class HypothesisEvidenceGraphReasoner(nn.Module):
    """Production graph reasoning for canonical HGT and controlled ablations."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.graph_type = str(config["type"]).lower()
        self.evidence_mode = str(config.get("evidence_mode", "heterogeneous")).lower()
        if self.graph_type not in {"hgt", "gatv2"}:
            raise ValueError("graph.type must be hgt or gatv2")
        if self.evidence_mode not in {"heterogeneous", "homogeneous"}:
            raise ValueError("graph.evidence_mode must be heterogeneous or homogeneous")
        if self.graph_type == "gatv2" and self.evidence_mode != "heterogeneous":
            raise ValueError("The GATv2 ablation cannot be combined with homogeneous_evidence")
        self.hidden_dim = int(config["hidden_dim"])
        self.cycles = int(config["reasoning"]["cycles"])
        if self.cycles < 1 or self.cycles > 3:
            raise ValueError("Graph reasoning cycles must be in the supported study range [1,3]")
        self.dynamic_rebuild = bool(config["reasoning"].get("dynamic_rebuild", False))
        if self.dynamic_rebuild and self.cycles < 2:
            raise ValueError("dynamic hypothesis refinement requires graph.reasoning.cycles >= 2")
        self.hypotheses = HypothesisNodeInitializer(config)
        self.builder = HypothesisEvidenceGraphBuilder(config)
        self.evidence_adapters = TypedEvidenceFeatureAdapters(
            hidden_dim=self.hidden_dim,
            input_dims=config.get("input_dims", {}),
        )
        encoder_kwargs = {
            "hidden_dim": self.hidden_dim,
            "layers": int(config["layers"]),
            "heads": int(config["heads"]),
            "dropout": float(config["dropout"]),
        }
        if self.graph_type == "gatv2":
            self.gatv2 = GATv2GraphEncoder(**encoder_kwargs)
            self._encoder_name = "gatv2"
        elif self.evidence_mode == "homogeneous":
            self.homogeneous_hgt = HomogeneousEvidenceHGTEncoder(**encoder_kwargs)
            self._encoder_name = "homogeneous_hgt"
        else:
            self.hgt = HGTGraphEncoder(**encoder_kwargs)
            self._encoder_name = "hgt"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HypothesisEvidenceGraphReasoner":
        """Build from a top-level YAML config."""

        return cls(config["graph"])

    def forward(self, evidence: HeterogeneousEvidenceBatch) -> HypothesisGraphOutput:
        """Return graph-derived FS/FA/TS/TA embeddings, not logits."""

        evidence = self.evidence_adapters(evidence)
        batch_size = evidence.visual_semantic.features.shape[0]
        initial_hypotheses = self.hypotheses(batch_size)
        current_evidence = evidence
        current_hypotheses = initial_hypotheses
        graph = None
        static_graph = None
        graph_build_count = 0
        x_dict: dict[str, torch.Tensor] = {}
        intermediate: list[torch.Tensor] = []
        cycle_statistics: list[dict[str, Any]] = []
        for cycle_index in range(self.cycles):
            if cycle_index == 0:
                graph = self.builder.build(current_evidence, current_hypotheses)
                static_graph = graph
                graph_build_count += 1
            elif self.dynamic_rebuild:
                current_evidence = _refresh_cross_modal_metadata(current_evidence)
                graph = self.builder.build(current_evidence, current_hypotheses)
                graph_build_count += 1
            else:
                if static_graph is None or not x_dict:
                    raise RuntimeError(
                        "Static multi-pass reasoning requires an initialized graph state"
                    )
                graph = _reuse_static_graph(
                    static_graph,
                    updated_node_features=x_dict,
                    cycle_index=cycle_index,
                )
            encoder = getattr(self, self._encoder_name)
            x_dict = encoder(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
            current_hypotheses = _unflatten_hypotheses(x_dict[HYPOTHESIS], batch_size=batch_size)
            intermediate.append(current_hypotheses)
            current_evidence = _refined_evidence_batch(current_evidence, graph, x_dict)
            cycle_statistics.append(graph_statistics(graph))
        if graph is None:
            raise RuntimeError("Graph reasoning produced no cycles")
        hypothesis_features = current_hypotheses
        output = HypothesisGraphOutput(
            hypothesis_features=hypothesis_features,
            hypothesis_mask=torch.ones(
                batch_size,
                len(HYPOTHESIS_ORDER),
                dtype=torch.bool,
                device=hypothesis_features.device,
            ),
            graph_batch=graph,
            graph_statistics={
                **graph_statistics(graph),
                "dynamic_rebuild": self.dynamic_rebuild,
                "rebuild_mode": _rebuild_mode(
                    cycles=self.cycles,
                    dynamic_rebuild=self.dynamic_rebuild,
                ),
                "graph_build_count": graph_build_count,
                "reliability_reassessment": False,
                "cycle_statistics": cycle_statistics,
            },
            updated_node_features=x_dict,
            intermediate_hypothesis_features=intermediate,
            refined_visual_semantic=current_evidence.visual_semantic,
            refined_visual_forensic=current_evidence.visual_forensic,
            refined_text_alignment=current_evidence.text_alignment,
            refined_text_context=current_evidence.text_context,
        )
        output.validate()
        return output

    def graph_encoder_module(self) -> nn.Module:
        """Return the active graph encoder for parameter accounting."""

        return getattr(self, self._encoder_name)


def _rebuild_mode(*, cycles: int, dynamic_rebuild: bool) -> str:
    if cycles == 1:
        return "none"
    return "state_dependent" if dynamic_rebuild else "static"


def _reuse_static_graph(
    graph: Any,
    *,
    updated_node_features: Mapping[str, torch.Tensor],
    cycle_index: int,
) -> Any:
    """Reuse frozen topology/edge attributes with the latest node states.

    This is the depth-only control used by the cycle study.  No graph builder,
    semantic re-query, edge selection, edge-strength update, or reliability
    update is performed after the first pass.
    """

    reused = replace(
        graph,
        x_dict=dict(updated_node_features),
        metadata={
            **dict(graph.metadata),
            "static_graph_reuse": True,
            "static_graph_cycle_index": int(cycle_index),
        },
    )
    reused.validate()
    return reused


def _unflatten_hypotheses(features: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if features.shape[0] != batch_size * len(HYPOTHESIS_ORDER):
        raise ValueError("Flattened hypothesis features must contain B*4 rows")
    return features.reshape(batch_size, len(HYPOTHESIS_ORDER), features.shape[-1])


def _refined_evidence_set(
    original: EvidenceSet,
    sample_indices: torch.Tensor,
    local_indices: torch.Tensor,
    refined_features: torch.Tensor,
) -> EvidenceSet:
    features = original.features.new_zeros(original.features.shape)
    if refined_features.numel():
        features[sample_indices, local_indices] = refined_features
    return replace(original, features=features)


def _refined_evidence_batch(
    evidence: HeterogeneousEvidenceBatch,
    graph: Any,
    x_dict: Mapping[str, torch.Tensor],
) -> HeterogeneousEvidenceBatch:
    """Reconstruct the typed evidence state after one hypothesis-feedback cycle."""

    return HeterogeneousEvidenceBatch(
        visual_semantic=_refined_evidence_set(
            evidence.visual_semantic,
            graph.node_sample_indices[VISUAL_SEMANTIC],
            graph.node_local_indices[VISUAL_SEMANTIC],
            x_dict[VISUAL_SEMANTIC],
        ),
        visual_forensic=_refined_evidence_set(
            evidence.visual_forensic,
            graph.node_sample_indices[VISUAL_FORENSIC],
            graph.node_local_indices[VISUAL_FORENSIC],
            x_dict[VISUAL_FORENSIC],
        ),
        text_alignment=_refined_evidence_set(
            evidence.text_alignment,
            graph.node_sample_indices[TEXT_ALIGNMENT],
            graph.node_local_indices[TEXT_ALIGNMENT],
            x_dict[TEXT_ALIGNMENT],
        ),
        text_context=_refined_evidence_set(
            evidence.text_context,
            graph.node_sample_indices[TEXT_CONTEXT],
            graph.node_local_indices[TEXT_CONTEXT],
            x_dict[TEXT_CONTEXT],
        ),
        cross_modal_metadata=evidence.cross_modal_metadata,
    )


def _refresh_cross_modal_metadata(evidence: HeterogeneousEvidenceBatch) -> HeterogeneousEvidenceBatch:
    """Re-query semantic edges from the refined evidence state before the next cycle."""

    visual = evidence.visual_semantic
    text = evidence.text_alignment
    visual_features = torch.nn.functional.normalize(visual.features.float(), dim=-1)
    text_features = torch.nn.functional.normalize(text.features.float(), dim=-1)
    similarity = torch.bmm(visual_features, text_features.transpose(1, 2)).to(visual.features.dtype)
    valid = visual.mask.unsqueeze(2) & text.mask.unsqueeze(1)
    similarity = similarity.masked_fill(~valid, 0.0)
    visual_reliability = (
        visual.mask.to(visual.features.dtype)
        if visual.reliability_scores is None
        else visual.reliability_scores
    )
    text_reliability = (
        text.mask.to(text.features.dtype)
        if text.reliability_scores is None
        else text.reliability_scores
    )
    confidence = torch.sqrt(
        (
            visual_reliability.float().unsqueeze(2).clamp_min(0.0)
            * text_reliability.float().unsqueeze(1).clamp_min(0.0)
        ).clamp_min(1e-12)
    ).to(similarity.dtype)
    confidence = confidence.masked_fill(~valid, 0.0)
    previous = evidence.cross_modal_metadata
    metadata = CrossModalMetadata(
        similarity=similarity,
        visual_mask=visual.mask,
        text_mask=text.mask,
        visual_source_indices=visual.source_indices,
        text_source_indices=text.source_indices,
        edge_confidence=confidence,
        alignment_loss=previous.alignment_loss,
        method="iterative_cosine_requery",
        feature_space="graph_refined_hidden",
    )
    return replace(evidence, cross_modal_metadata=metadata)
