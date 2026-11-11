"""Full HERA-DGM production model composition."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from heri.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor
from heri.experiments import get_hera_ablation_variant, get_hera_cycle_study_variant
from heri.graph import HypothesisEvidenceGraphReasoner
from heri.graph.query_reasoner import LearnedQueryReasoner
from heri.models.baselines.base import BaselineBackboneMixin, finite_or_fail
from heri.models.baselines.factory import build_backbones
from heri.models.outputs import BackboneOutputs
from heri.routing import EvidenceReliabilityRouter, RoutingDiagnostics, summarize_routed_batch

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
        self.ablation_variant = get_hera_ablation_variant(config)
        self.cycle_study_variant = get_hera_cycle_study_variant(config)
        self.cycle_study_protocol_id = config.get("cycle_study", {}).get(
            "protocol_id"
        )
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
        if not bool(getattr(self.router, "enabled", True)):
            self.router.requires_grad_(False)
        self.graph_reasoner = (
            graph_reasoner
            if graph_reasoner is not None
            else (LearnedQueryReasoner(config)
                  if config.get("reasoning_control", {}).get("enabled", False)
                  else HypothesisEvidenceGraphReasoner.from_config(config))
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
                "image_grounding_boxes": output.image_grounding_boxes,
                "text_grounding_logits": output.text_grounding_logits,
            }
        )
        output.validate()
        return output

    def new_modules(self) -> list[nn.Module]:
        """Return all trainable non-backbone HERA modules for optimizer grouping."""

        modules = [self.evidence_extractor, self.graph_reasoner, self.decision]
        if bool(getattr(self.router, "enabled", True)):
            modules.insert(1, self.router)
        return modules

    def hera_component_modules(self) -> dict[str, nn.Module]:
        """Return component groups for parameter audits."""

        modules: dict[str, nn.Module] = {
            "evidence_extractor": self.evidence_extractor,
            "router": self.router,
            "graph_reasoner": self.graph_reasoner,
            "decision": self.decision,
        }
        if hasattr(self.graph_reasoner, "graph_encoder_module"):
            graph_encoder = self.graph_reasoner.graph_encoder_module()
            if (
                getattr(self.graph_reasoner, "graph_type", None) == "hgt"
                and getattr(self.graph_reasoner, "evidence_mode", None) == "heterogeneous"
            ):
                modules["hgt"] = graph_encoder
            else:
                modules["graph_encoder"] = graph_encoder
        elif hasattr(self.graph_reasoner, "hgt"):
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
            if class_name in {"binaryhead", "independenthypothesisbinaryhead"} or terminal_name in {
                "binary_head",
                "binary_classifier",
            }:
                count += 1
        return count

    def assert_no_binary_shortcut(self) -> None:
        """Fail if a direct Real/Fake classifier appears in the HERA module tree."""

        shortcut_count = self.count_independent_binary_shortcuts()
        aggregation = self.config.get("model", {}).get("binary", {}).get("aggregation", "noisy_or")
        expected = 1 if aggregation == "independent_head" else 0
        if shortcut_count != expected:
            raise ValueError(
                "HERA binary shortcut count violates the experiment contract: "
                f"variant={self.ablation_variant!r}, expected={expected}, found={shortcut_count}"
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
        graph_label = {
            "gatv2": "GATv2",
            "homogeneous_evidence": "HomogeneousEvidenceHGT",
        }.get(self.ablation_variant, "HGT")
        if self.config.get("reasoning_control", {}).get("enabled", False):
            graph_label = "LearnedQueries"
        elif self.config.get("graph", {}).get("reasoning", {}).get("operator") == "pooled":
            graph_label = "PooledEvidenceReadout"
        elif not self.config.get("graph", {}).get("include_hypothesis_nodes", True):
            graph_label = "EvidenceOnlyGraphReadout"
        router_label = (
            "EvidenceReliabilityRouter" if bool(getattr(self.router, "enabled", True))
            else "UniformEvidenceBypass"
        )
        binary_label = (
            "IndependentHypothesisBinaryHead"
            if self.config.get("model", {}).get("binary", {}).get("aggregation") == "independent_head"
            else "StableNoisyOR"
        )
        metadata.update(
            {
                "model_type": "hera_dgm",
                "architecture_version": self.config["project"]["architecture_version"],
                "ablation_variant": self.ablation_variant,
                "cycle_study_variant": self.cycle_study_variant,
                "cycle_study_protocol_id": self.cycle_study_protocol_id,
                "pipeline": [
                    "SigLIP2",
                    "DINOv3",
                    "DeBERTa",
                    "HeterogeneousEvidenceExtractor",
                    router_label,
                    graph_label,
                    "HERADecisionModule",
                    binary_label,
                ],
                "router_family_mean_reliability": _router_family_mean_reliability(routed),
                "router_batch_summary": _router_batch_summary(self.config, routed),
                "forward_forbidden_target_keys": sorted(FORWARD_FORBIDDEN_TARGET_KEYS),
                "dynamic_graph_rebuild": bool(
                    self.config["graph"]["reasoning"].get("dynamic_rebuild", False)
                ),
                "evidence_sufficiency_protocol": self.config.get("loss", {})
                .get("evidence_sufficiency", {})
                .get("protocol", "per_modality_v1"),
                "evidence_sufficiency_aggregation": self.config.get("loss", {})
                .get("evidence_sufficiency", {})
                .get("aggregation", "per_modality"),
                "evidence_support_summary": "minimum_applicable_modality",
                "paper_contract": "HERI",
            }
        )
        if not self.training:
            with torch.no_grad():
                metadata["hypothesis_probabilities_by_cycle"] = [
                    torch.sigmoid(self.decision.attribution_head(state).float()).cpu().tolist()
                    for state in graph_output.intermediate_hypothesis_features
                ]
        return HERADGMOutput(
            atomic_logits=decision_output.atomic_logits,
            atomic_probabilities=decision_output.atomic_probabilities,
            binary_probability=decision_output.binary_probability,
            binary_logit=decision_output.binary_logit,
            image_grounding_logits=decision_output.image_grounding_logits,
            image_grounding_probabilities=decision_output.image_grounding_probabilities,
            image_grounding_mask=decision_output.image_grounding_mask,
            image_grounding_boxes=decision_output.image_grounding_boxes,
            image_grounding_patch_boxes=decision_output.image_grounding_patch_boxes,
            text_grounding_logits=decision_output.text_grounding_logits,
            text_grounding_probabilities=decision_output.text_grounding_probabilities,
            text_grounding_mask=decision_output.text_grounding_mask,
            hypothesis_features=decision_output.hypothesis_features,
            image_source_indices=decision_output.image_source_indices,
            text_word_ids=decision_output.text_word_ids,
            evidence_support=decision_output.evidence_support,
            evidence_support_by_modality=(
                decision_output.evidence_support_by_modality
            ),
            evidence_support_modality_mask=(
                decision_output.evidence_support_modality_mask
            ),
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
                "graph_type": getattr(self.graph_reasoner, "graph_type", str(self.config["graph"]["type"])),
                "evidence_mode": str(self.config["graph"].get("evidence_mode", "heterogeneous")),
                "hypothesis_feedback": bool(
                    self.config["graph"]["reasoning"]["hypothesis_feedback"]
                ) and not self.config.get("reasoning_control", {}).get("enabled", False),
                "rebuild_mode": graph_output.graph_statistics.get("rebuild_mode"),
                "graph_build_count": graph_output.graph_statistics.get("graph_build_count"),
                "reliability_reassessment": graph_output.graph_statistics.get(
                    "reliability_reassessment"
                ),
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
