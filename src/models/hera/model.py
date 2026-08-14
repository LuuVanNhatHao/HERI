"""Full Phase-7 HERA-DGM production model composition."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor
from src.graph import HypothesisEvidenceGraphReasoner
from src.models.baselines.base import BaselineBackboneMixin, finite_or_fail
from src.models.baselines.factory import build_backbones
from src.models.outputs import BackboneOutputs
from src.routing import EvidenceReliabilityRouter, RoutingDiagnostics, summarize_routed_batch

from .decision import HERADecisionModule
from .outputs import HERADecisionOutput, HERADGMOutput


FORWARD_FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "binary_target",
        "atomic_target",
        "fake_cls",
        "fake_image_box_target",
        "manipulated_face_target_index",
        "fake_text_pos",
    }
)


class HERADGM(nn.Module, BaselineBackboneMixin):
    """Canonical HERA-DGM model: evidence -> attribution -> Noisy-OR authenticity."""

    model_type = "hera"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        siglip2: nn.Module | None = None,
        dinov3: nn.Module | None = None,
        deberta: nn.Module | None = None,
        evidence_extractor: nn.Module | None = None,
        router: nn.Module | None = None,
        graph_reasoner: nn.Module | None = None,
        decision_module: nn.Module | None = None,
        load_backbones: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        backbones = build_backbones(config) if load_backbones else {}
        self.siglip2 = siglip2 if siglip2 is not None else backbones.get("siglip2")
        self.dinov3 = dinov3 if dinov3 is not None else backbones.get("dinov3")
        self.deberta = deberta if deberta is not None else backbones.get("deberta")
        self.evidence_extractor = (
            evidence_extractor
            if evidence_extractor is not None
            else HeterogeneousEvidenceExtractor.from_config(config)
        )
        self.router = router if router is not None else EvidenceReliabilityRouter.from_config(config)
        self.graph_reasoner = (
            graph_reasoner
            if graph_reasoner is not None
            else HypothesisEvidenceGraphReasoner.from_config(config)
        )
        self.decision = (
            decision_module
            if decision_module is not None
            else HERADecisionModule.from_config(config)
        )
        self.assert_no_binary_shortcut()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, load_backbones: bool = True) -> "HERADGM":
        """Build the full HERA-DGM model from YAML-derived configuration."""

        return cls(config, load_backbones=load_backbones)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        features: BackboneOutputs | None = None,
    ) -> HERADGMOutput:
        """Run the target-independent full HERA-DGM forward path."""

        encoded = self._encode_backbones(batch, features)
        evidence = self.evidence_extractor(encoded, EvidenceStructuralMetadata.from_batch(batch))
        routed = self.router(evidence)
        graph_output = self.graph_reasoner(routed.as_graph_evidence())
        decision_output = self.decision(graph_output)
        output = self._to_hera_output(
            decision_output=decision_output,
            evidence=evidence,
            routed=routed,
            graph_output=graph_output,
        )
        finite_or_fail(
            {
                "atomic_logits": output.atomic_logits,
                "binary_logit": output.binary_logit,
                "image_grounding_logits": output.image_grounding_logits,
                "text_grounding_logits": output.text_grounding_logits,
            }
        )
        output.validate()
        return output

    def new_modules(self) -> list[nn.Module]:
        """Return all trainable non-backbone HERA modules for optimizer grouping."""

        return [self.evidence_extractor, self.router, self.graph_reasoner, self.decision]

    def hera_component_modules(self) -> dict[str, nn.Module]:
        """Return Phase-7 component groups for parameter audits."""

        modules: dict[str, nn.Module] = {
            "evidence_extractor": self.evidence_extractor,
            "router": self.router,
            "graph_reasoner": self.graph_reasoner,
            "decision": self.decision,
        }
        if hasattr(self.graph_reasoner, "hgt"):
            modules["hgt"] = self.graph_reasoner.hgt
        if hasattr(self.graph_reasoner, "hypotheses"):
            modules["hypothesis_embeddings"] = self.graph_reasoner.hypotheses
        if hasattr(self.decision, "attribution_head"):
            modules["attribution_head"] = self.decision.attribution_head
        if hasattr(self.decision, "image_grounding_head"):
            modules["image_grounding"] = self.decision.image_grounding_head
        if hasattr(self.decision, "text_grounding_head"):
            modules["text_grounding"] = self.decision.text_grounding_head
        return modules

    def count_independent_binary_shortcuts(self) -> int:
        """Count forbidden direct binary classifier modules in canonical HERA."""

        count = 0
        for module_name, module in self.named_modules():
            if not module_name:
                continue
            terminal_name = module_name.rsplit(".", 1)[-1].lower()
            class_name = module.__class__.__name__.lower()
            if class_name == "binaryhead" or terminal_name in {"binary_head", "binary_classifier"}:
                count += 1
        return count

    def assert_no_binary_shortcut(self) -> None:
        """Fail if a direct Real/Fake classifier appears in the HERA module tree."""

        shortcut_count = self.count_independent_binary_shortcuts()
        if shortcut_count:
            raise ValueError(
                f"Canonical HERADGM must not contain an independent binary head; found {shortcut_count}"
            )

    def _to_hera_output(
        self,
        *,
        decision_output: HERADecisionOutput,
        evidence: Any,
        routed: Any,
        graph_output: Any,
    ) -> HERADGMOutput:
        metadata = dict(decision_output.metadata)
        metadata.update(
            {
                "model_type": "hera_dgm",
                "pipeline": [
                    "SigLIP2",
                    "DINOv3",
                    "DeBERTa",
                    "HeterogeneousEvidenceExtractor",
                    "EvidenceReliabilityRouter",
                    "HGT",
                    "HERADecisionModule",
                    "StableNoisyOR",
                ],
                "router_family_mean_reliability": _router_family_mean_reliability(routed),
                "router_batch_summary": _router_batch_summary(self.config, routed),
                "forward_forbidden_target_keys": sorted(FORWARD_FORBIDDEN_TARGET_KEYS),
            }
        )
        return HERADGMOutput(
            atomic_logits=decision_output.atomic_logits,
            atomic_probabilities=decision_output.atomic_probabilities,
            binary_probability=decision_output.binary_probability,
            binary_logit=decision_output.binary_logit,
            image_grounding_logits=decision_output.image_grounding_logits,
            image_grounding_probabilities=decision_output.image_grounding_probabilities,
            image_grounding_mask=decision_output.image_grounding_mask,
            text_grounding_logits=decision_output.text_grounding_logits,
            text_grounding_probabilities=decision_output.text_grounding_probabilities,
            text_grounding_mask=decision_output.text_grounding_mask,
            hypothesis_features=decision_output.hypothesis_features,
            face_source_indices=decision_output.face_source_indices,
            text_word_ids=decision_output.text_word_ids,
            evidence_support=decision_output.evidence_support,
            metadata=metadata,
            evidence_metadata={
                "families": {
                    name: {
                        "feature_shape": list(evidence_set.features.shape),
                        "valid_nodes_per_sample": [
                            int(value) for value in evidence_set.mask.sum(dim=1).detach().cpu().tolist()
                        ],
                    }
                    for name, evidence_set in evidence.families().items()
                }
            },
            routing_metadata=dict(routed.metadata),
            graph_metadata={
                "statistics": dict(graph_output.graph_statistics),
                "reasoning_cycles": len(graph_output.intermediate_hypothesis_features),
            },
            cross_modal_alignment_loss=evidence.cross_modal_metadata.alignment_loss,
            router_regularization_loss=routed.regularization_loss,
        )


def _router_family_mean_reliability(routed: Any) -> dict[str, list[float]]:
    summary: dict[str, list[float]] = {}
    for name, routed_set in routed.families().items():
        scores = routed_set.reliability_scores.detach()
        mask = routed_set.mask.detach().to(scores.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        means = (scores * mask).sum(dim=1) / denom
        summary[name] = [float(value) for value in means.cpu().tolist()]
    return summary


def _router_batch_summary(config: Mapping[str, Any], routed: Any) -> dict[str, Any] | None:
    diagnostics_config = config.get("routing", {}).get("diagnostics", {})
    if not diagnostics_config:
        return None
    diagnostics = RoutingDiagnostics(
        low_threshold=float(diagnostics_config["low_threshold"]),
        high_threshold=float(diagnostics_config["high_threshold"]),
    )
    return summarize_routed_batch(routed, diagnostics=diagnostics)
