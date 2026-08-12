"""Target-independent evidence reliability router."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.evidence.outputs import EvidenceSet, HeterogeneousEvidenceBatch

from .outputs import RoutedEvidenceBatch, RoutedEvidenceSet
from .reliability import EvidenceReliabilityEstimator


ROUTED_FAMILIES = ("visual_semantic", "visual_forensic", "face", "text_alignment", "text_context")


class EvidenceReliabilityRouter(nn.Module):
    """Estimate per-family reliability and gate evidence before HGT."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        routing_config = dict(config["routing"] if "routing" in config else config)
        self.config = routing_config
        self.enabled = bool(routing_config["enabled"])
        self.mode = str(routing_config["mode"])
        if self.mode not in {"soft", "hard"}:
            raise ValueError("routing.mode must be 'soft' or 'hard'")
        self.hard_enabled = bool(routing_config["hard_eval"]["enabled"])
        self.hard_threshold = float(routing_config["hard_eval"]["threshold"])
        if self.mode == "hard" and not self.hard_enabled:
            raise ValueError("routing.mode='hard' requires routing.hard_eval.enabled=true")
        if self.hard_threshold < 0 or self.hard_threshold > 1:
            raise ValueError("routing.hard_eval.threshold must be in [0,1]")
        estimator_config = routing_config["estimator"]
        self.feature_dim = int(routing_config["feature_dim"])
        self.family_enabled = {
            family: bool(routing_config[family]["enabled"])
            for family in ROUTED_FAMILIES
        }
        self.estimators = nn.ModuleDict(
            {
                family: EvidenceReliabilityEstimator(
                    input_dim=self.feature_dim,
                    hidden_dim=int(estimator_config["hidden_dim"]),
                    dropout=float(estimator_config["dropout"]),
                    initial_bias=float(estimator_config["initial_bias"]),
                    output_weight_std=float(estimator_config["output_weight_std"]),
                )
                for family in ROUTED_FAMILIES
            }
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EvidenceReliabilityRouter":
        """Build router from top-level YAML config."""

        return cls(config)

    def forward(self, evidence: HeterogeneousEvidenceBatch) -> RoutedEvidenceBatch:
        """Route Phase-3 evidence without receiving labels or targets."""

        evidence.validate()
        routed_sets = {
            family: self._route_set(family, evidence.families()[family])
            for family in ROUTED_FAMILIES
        }
        output = RoutedEvidenceBatch(
            visual_semantic=routed_sets["visual_semantic"],
            visual_forensic=routed_sets["visual_forensic"],
            face=routed_sets["face"],
            text_alignment=routed_sets["text_alignment"],
            text_context=routed_sets["text_context"],
            cross_modal_metadata=evidence.cross_modal_metadata,
            original=evidence,
            metadata={
                "enabled": self.enabled,
                "mode": self.mode,
                "hard_eval_enabled": self.hard_enabled,
                "hard_threshold": self.hard_threshold,
                "routing_position": "phase3_evidence_to_phase4_graph",
            },
        )
        output.validate()
        return output

    def _route_set(self, family: str, evidence_set: EvidenceSet) -> RoutedEvidenceSet:
        if evidence_set.features.shape[-1] != self.feature_dim:
            raise ValueError(f"{family} feature dim must equal routing.feature_dim={self.feature_dim}")
        features = evidence_set.features
        mask = evidence_set.mask.to(torch.bool)
        if not self.enabled or not self.family_enabled[family]:
            logits = features.new_zeros(mask.shape)
            scores = mask.to(features.dtype)
            routed_features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))
            effective_mask = mask
        else:
            logits = self.estimators[family](features)
            scores = torch.sigmoid(logits) * mask.to(logits.dtype)
            if self.mode == "hard":
                effective_mask = mask & (scores >= self.hard_threshold)
                routed_features = features * effective_mask.to(features.dtype).unsqueeze(-1)
            else:
                effective_mask = mask
                routed_features = features * scores.unsqueeze(-1).to(features.dtype)
            logits = logits.masked_fill(~mask, 0.0)
            routed_features = routed_features.masked_fill(~mask.unsqueeze(-1), 0.0)
        return RoutedEvidenceSet(
            family=family,
            original_features=features,
            routed_features=routed_features,
            mask=mask,
            reliability_logits=logits,
            reliability_scores=scores,
            source_indices=evidence_set.source_indices,
            provenance=evidence_set.provenance,
            original=evidence_set,
            effective_mask=effective_mask,
        )
