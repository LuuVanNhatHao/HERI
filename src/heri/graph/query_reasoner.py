"""Learned class-query control: routed evidence, without graph message passing."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .adapters import TypedEvidenceFeatureAdapters
from .hypotheses import HypothesisNodeInitializer
from .outputs import GraphBatch, HypothesisGraphOutput
from .schema import HYPOTHESIS, HypothesisSpec, edge_types_for_hypotheses


class LearnedQueryReasoner(nn.Module):
    """Reuse one decoder stack twice, matching HERA's shared-stack pass count.

    Evidence remains fixed. Query self-attention and cross-attention use no
    relation types, edge weights, graph rebuild, or evidence feedback.
    The empty-edge GraphBatch is an output compatibility container only.
    """

    graph_type = "learned_queries"
    evidence_mode = "heterogeneous"

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        cfg = dict(config["graph"])
        cfg["hypotheses"] = dict(
            config.get("model", {}).get(
                "hypotheses",
                {"order": ["FS", "FA", "TS", "TA"], "image": ["FS", "FA"], "text": ["TS", "TA"]},
            )
        )
        self.hidden_dim = int(cfg["hidden_dim"])
        self.cycles = int(cfg["reasoning"]["cycles"])
        self.hypotheses = HypothesisNodeInitializer(cfg)
        self.hypothesis_spec = HypothesisSpec.from_mapping(cfg.get("hypotheses"))
        self.edge_types = edge_types_for_hypotheses(self.hypothesis_spec.order)
        self.evidence_adapters = TypedEvidenceFeatureAdapters(
            hidden_dim=self.hidden_dim, input_dims=cfg.get("input_dims", {})
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim, nhead=int(cfg["heads"]),
                dim_feedforward=4 * self.hidden_dim, dropout=float(cfg["dropout"]),
                activation="gelu", batch_first=True,
            ), num_layers=int(cfg["layers"]),
        )

    def graph_encoder_module(self) -> nn.Module:
        return self.decoder

    def forward(self, evidence):
        evidence = self.evidence_adapters(evidence)
        families = {name: getattr(evidence, name) for name in (
            "visual_semantic", "visual_forensic", "text_alignment", "text_context"
        )}
        memory = torch.cat([item.features for item in families.values()], dim=1)
        mask = torch.cat([item.mask for item in families.values()], dim=1)
        # A fixed zero sentinel prevents NaNs for an entirely empty sample.
        memory = torch.cat([memory, memory.new_zeros(memory.shape[0], 1, self.hidden_dim)], dim=1)
        mask = torch.cat([mask, ~mask.any(dim=1, keepdim=True)], dim=1)
        batch_size = memory.shape[0]
        state = self.hypotheses(batch_size)
        intermediate = []
        for _ in range(self.cycles):
            state = self.decoder(state, memory, memory_key_padding_mask=~mask)
            intermediate.append(state)
        x_dict, sample_indices, local_indices = {}, {}, {}
        for name, item in families.items():
            sample, local = item.mask.nonzero(as_tuple=True)
            x_dict[name] = item.features[sample, local]
            sample_indices[name], local_indices[name] = sample, local
        x_dict[HYPOTHESIS] = state.reshape(-1, self.hidden_dim)
        hypothesis_count = len(self.hypothesis_spec.order)
        sample_indices[HYPOTHESIS] = torch.arange(batch_size, device=state.device).repeat_interleave(hypothesis_count)
        local_indices[HYPOTHESIS] = torch.arange(hypothesis_count, device=state.device).repeat(batch_size)
        graph = GraphBatch(
            x_dict=x_dict, node_sample_indices=sample_indices, node_local_indices=local_indices,
            edge_index_dict={edge: torch.empty(2, 0, dtype=torch.long, device=state.device) for edge in self.edge_types},
            edge_sample_indices={edge: torch.empty(0, dtype=torch.long, device=state.device) for edge in self.edge_types},
            edge_attr_dict={edge: state.new_empty(0, 2) for edge in self.edge_types},
            batch_size=batch_size, hidden_dim=self.hidden_dim,
            metadata={"container_only": True, "reasoning_backend": self.graph_type,
                      "hypothesis_order": list(self.hypothesis_spec.order),
                      "edge_types": list(self.edge_types)},
        )
        output = HypothesisGraphOutput(
            hypothesis_features=state, hypothesis_mask=torch.ones(batch_size, hypothesis_count, dtype=torch.bool, device=state.device),
            graph_batch=graph, updated_node_features=x_dict,
            graph_statistics={"reasoning_backend": self.graph_type, "graph_edges_used": False,
                              "graph_build_count": 0, "dynamic_rebuild": False, "rebuild_mode": "none", "reliability_reassessment": False},
            intermediate_hypothesis_features=intermediate,
            **{f"refined_{name}": item for name, item in families.items()},
        )
        output.validate()
        return output
