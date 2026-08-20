"""Canonical Phase-5 HERA decision module."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.graph.outputs import HypothesisGraphOutput
from src.graph.schema import HYPOTHESIS_ORDER
from src.models.heads.binary import IndependentHypothesisBinaryHead
from src.models.heads.attribution import HypothesisAttributionHead
from src.models.heads.hypothesis_grounding import (
    GenericImageGroundingHead,
    GenericTextGroundingHead,
    HypothesisConditionedImageGroundingHead,
    HypothesisConditionedTextGroundingHead,
)
from src.models.heads.noisy_or import StableNoisyOR

from .outputs import HERADecisionOutput


class HERADecisionModule(nn.Module):
    """Convert graph-derived hypotheses to attribution, grounding, and Noisy-OR verdict."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        model_config = dict(config["model"] if "model" in config else config)
        order = tuple(model_config["hypotheses"]["order"])
        if order != HYPOTHESIS_ORDER:
            raise ValueError(f"model.hypotheses.order must be {list(HYPOTHESIS_ORDER)}, got {list(order)}")
        hidden_dim = int(model_config.get("hidden_dim", model_config["attribution"]["hidden_dim"]))
        if hidden_dim != int(model_config["attribution"]["hidden_dim"]):
            raise ValueError("model.hidden_dim and model.attribution.hidden_dim must match")
        self.hidden_dim = hidden_dim
        self.hypothesis_order = HYPOTHESIS_ORDER

        attribution_config = model_config["attribution"]
        image_config = model_config["grounding"]["image"]
        text_config = model_config["grounding"]["text"]
        binary_config = model_config["binary"]
        self.grounding_mode = str(model_config["grounding"].get("mode", "hypothesis_conditioned"))
        self.grounding_routing = str(model_config["grounding"].get("routing", "soft"))
        self.grounding_routing_temperature = float(
            model_config["grounding"].get("routing_temperature", 1.0)
        )
        if self.grounding_routing not in {"soft", "hard"}:
            raise ValueError("model.grounding.routing must be soft or hard")
        if self.grounding_routing_temperature <= 0:
            raise ValueError("model.grounding.routing_temperature must be positive")
        if self.grounding_mode not in {"hypothesis_conditioned", "generic"}:
            raise ValueError("model.grounding.mode must be hypothesis_conditioned or generic")
        self.binary_aggregation = str(binary_config["aggregation"])
        if self.binary_aggregation not in {"noisy_or", "independent_head"}:
            raise ValueError("model.binary.aggregation must be noisy_or or independent_head")
        if not bool(text_config["full_token_store"]):
            raise ValueError("Phase-5 text grounding requires model.grounding.text.full_token_store=true")

        self.attribution_head = HypothesisAttributionHead(
            hidden_dim=hidden_dim,
            dropout=float(attribution_config["dropout"]),
        )
        self.noisy_or = StableNoisyOR()
        if self.grounding_mode == "generic":
            self.image_grounding_head = GenericImageGroundingHead(
                hidden_dim=hidden_dim,
                dropout=float(image_config["dropout"]),
            )
            self.text_grounding_head = GenericTextGroundingHead(
                hidden_dim=hidden_dim,
                token_input_dim=int(text_config["token_input_dim"]),
                dropout=float(text_config["dropout"]),
            )
        else:
            self.image_grounding_head = HypothesisConditionedImageGroundingHead(
                hidden_dim=hidden_dim,
                dropout=float(image_config["dropout"]),
                scoring=str(image_config["scoring"]),
            )
            self.text_grounding_head = HypothesisConditionedTextGroundingHead(
                hidden_dim=hidden_dim,
                token_input_dim=int(text_config["token_input_dim"]),
                dropout=float(text_config["dropout"]),
                scoring=str(text_config["scoring"]),
            )
        if self.binary_aggregation == "independent_head":
            self.binary_head = IndependentHypothesisBinaryHead(
                hidden_dim=hidden_dim,
                hypotheses=len(HYPOTHESIS_ORDER),
                dropout=float(attribution_config["dropout"]),
            )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HERADecisionModule":
        """Build the decision module from the top-level YAML config."""

        return cls(config)

    def forward(self, graph_output: HypothesisGraphOutput) -> HERADecisionOutput:
        """Forward pass using graph/evidence only; no targets are accepted."""

        graph_output.validate()
        if graph_output.refined_visual_forensic is None:
            raise ValueError("refined_visual_forensic is required for dense image grounding")
        if graph_output.refined_text_context is None:
            raise ValueError("HypothesisGraphOutput.refined_text_context is required for Phase-5 full-text grounding")
        atomic_logits = self.attribution_head(
            graph_output.hypothesis_features,
            graph_output.hypothesis_mask,
        )
        noisy_or_output = self.noisy_or(atomic_logits)
        if self.binary_aggregation == "independent_head":
            binary_logit = self.binary_head(graph_output.hypothesis_features)
            binary_probability = torch.sigmoid(binary_logit)
        else:
            binary_logit = noisy_or_output.binary_logit
            binary_probability = noisy_or_output.binary_probability
        image_logits, image_probabilities, image_mask, image_boxes = self.image_grounding_head(
            hypothesis_features=graph_output.hypothesis_features,
            image_evidence=graph_output.refined_visual_forensic,
        )
        text_logits, text_probabilities, text_mask = self.text_grounding_head(
            hypothesis_features=graph_output.hypothesis_features,
            text_context=graph_output.refined_text_context,
        )
        evidence_support = torch.cat(
            [
                _absolute_grounding_support(image_logits, image_mask),
                _absolute_grounding_support(text_logits, text_mask),
            ],
            dim=1,
        )
        output = HERADecisionOutput(
            atomic_logits=atomic_logits,
            atomic_probabilities=noisy_or_output.atomic_probabilities,
            binary_probability=binary_probability,
            binary_logit=binary_logit,
            image_grounding_logits=image_logits,
            image_grounding_probabilities=image_probabilities,
            image_grounding_mask=image_mask,
            image_grounding_boxes=image_boxes,
            image_grounding_patch_boxes=graph_output.refined_visual_forensic.full_normalized_boxes,
            text_grounding_logits=text_logits,
            text_grounding_probabilities=text_probabilities,
            text_grounding_mask=text_mask,
            hypothesis_features=graph_output.hypothesis_features,
            image_source_indices=graph_output.refined_visual_forensic.full_source_indices,
            text_word_ids=graph_output.refined_text_context.full_word_ids,
            metadata={
                "hypothesis_order": list(HYPOTHESIS_ORDER),
                "image_grounding_order": ["FS", "FA"],
                "text_grounding_order": ["TS", "TA"],
                "binary_aggregation": self.binary_aggregation,
                "binary_shortcut": self.binary_aggregation == "independent_head",
                "grounding_mode": self.grounding_mode,
                "grounding_routing": self.grounding_routing,
                "grounding_routing_temperature": self.grounding_routing_temperature,
                "image_grounding_protocol": "dense_bbox_from_raw_image",
                "annotation_mtcnn_boxes_used": False,
            },
            evidence_support=evidence_support,
        )
        output.validate()
        return output

    def backbone_modules(self) -> dict[str, nn.Module]:
        """Phase 5 attaches no pretrained backbone modules directly."""

        return {}

    def new_modules(self) -> list[nn.Module]:
        """Return trainable non-backbone modules for future optimizer grouping."""

        modules = [self.attribution_head, self.image_grounding_head, self.text_grounding_head]
        if hasattr(self, "binary_head"):
            modules.append(self.binary_head)
        return modules


def _absolute_grounding_support(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return absolute (non-softmax) local support for two hypotheses."""

    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError("grounding logits must be [B,2,N]")
    if mask.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("grounding mask must be [B,N]")
    if logits.shape[-1] == 0:
        return logits.new_zeros((logits.shape[0], 2))
    probabilities = torch.sigmoid(logits).masked_fill(~mask.unsqueeze(1), 0.0)
    return probabilities.max(dim=-1).values
