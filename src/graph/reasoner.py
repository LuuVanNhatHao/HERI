"""Phase-4 hypothesis-evidence graph reasoner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import torch
from torch import nn

from src.evidence.outputs import EvidenceSet, HeterogeneousEvidenceBatch

from .adapters import TypedEvidenceFeatureAdapters
from .builder import HypothesisEvidenceGraphBuilder
from .hgt import HGTGraphEncoder
from .hypotheses import HypothesisNodeInitializer
from .outputs import HypothesisGraphOutput
from .schema import HYPOTHESIS, HYPOTHESIS_ORDER
from .schema import FACE, TEXT_ALIGNMENT, TEXT_CONTEXT, VISUAL_FORENSIC, VISUAL_SEMANTIC
from .validation import graph_statistics


class HypothesisEvidenceGraphReasoner(nn.Module):
    """Real HGT-based reasoning from evidence nodes to hypothesis states."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        if str(config["type"]).lower() != "hgt":
            raise ValueError("Phase-4 canonical graph.type must be hgt")
        self.config = dict(config)
        self.hidden_dim = int(config["hidden_dim"])
        self.cycles = int(config["reasoning"]["cycles"])
        if self.cycles < 1 or self.cycles > 2:
            raise ValueError("Phase-4 reasoning cycles must be 1 or 2")
        self.hypotheses = HypothesisNodeInitializer(config)
        self.builder = HypothesisEvidenceGraphBuilder(config)
        self.evidence_adapters = TypedEvidenceFeatureAdapters(
            hidden_dim=self.hidden_dim,
            input_dims=config.get("input_dims", {}),
        )
        self.hgt = HGTGraphEncoder(
            hidden_dim=self.hidden_dim,
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            dropout=float(config["dropout"]),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HypothesisEvidenceGraphReasoner":
        """Build from a top-level YAML config."""

        return cls(config["graph"])

    def forward(self, evidence: HeterogeneousEvidenceBatch) -> HypothesisGraphOutput:
        """Return graph-derived FS/FA/TS/TA embeddings, not logits."""

        evidence = self.evidence_adapters(evidence)
        batch_size = evidence.visual_semantic.features.shape[0]
        initial_hypotheses = self.hypotheses(batch_size)
        graph = self.builder.build(evidence, initial_hypotheses)
        x_dict = dict(graph.x_dict)
        intermediate: list[torch.Tensor] = []
        for _ in range(self.cycles):
            x_dict = self.hgt(x_dict, graph.edge_index_dict)
            intermediate.append(_unflatten_hypotheses(x_dict[HYPOTHESIS], batch_size=batch_size))
        hypothesis_features = intermediate[-1]
        output = HypothesisGraphOutput(
            hypothesis_features=hypothesis_features,
            hypothesis_mask=torch.ones(
                batch_size,
                len(HYPOTHESIS_ORDER),
                dtype=torch.bool,
                device=hypothesis_features.device,
            ),
            graph_batch=graph,
            graph_statistics=graph_statistics(graph),
            updated_node_features=x_dict,
            intermediate_hypothesis_features=intermediate,
            refined_visual_semantic=_refined_evidence_set(
                evidence.visual_semantic,
                graph.node_sample_indices[VISUAL_SEMANTIC],
                graph.node_local_indices[VISUAL_SEMANTIC],
                x_dict[VISUAL_SEMANTIC],
            ),
            refined_visual_forensic=_refined_evidence_set(
                evidence.visual_forensic,
                graph.node_sample_indices[VISUAL_FORENSIC],
                graph.node_local_indices[VISUAL_FORENSIC],
                x_dict[VISUAL_FORENSIC],
            ),
            refined_face=_refined_evidence_set(
                evidence.face,
                graph.node_sample_indices[FACE],
                graph.node_local_indices[FACE],
                x_dict[FACE],
            ),
            refined_text_alignment=_refined_evidence_set(
                evidence.text_alignment,
                graph.node_sample_indices[TEXT_ALIGNMENT],
                graph.node_local_indices[TEXT_ALIGNMENT],
                x_dict[TEXT_ALIGNMENT],
            ),
            refined_text_context=_refined_evidence_set(
                evidence.text_context,
                graph.node_sample_indices[TEXT_CONTEXT],
                graph.node_local_indices[TEXT_CONTEXT],
                x_dict[TEXT_CONTEXT],
            ),
        )
        output.validate()
        return output


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
