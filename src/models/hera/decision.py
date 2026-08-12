"""Canonical Phase-5 HERA decision module."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from src.graph.outputs import HypothesisGraphOutput
from src.graph.schema import HYPOTHESIS_ORDER
from src.models.heads.attribution import HypothesisAttributionHead
from src.models.heads.hypothesis_grounding import (
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
        if str(binary_config["aggregation"]) != "noisy_or":
            raise ValueError("Canonical HERA binary aggregation must be noisy_or")
        if not bool(text_config["full_token_store"]):
            raise ValueError("Phase-5 text grounding requires model.grounding.text.full_token_store=true")

        self.attribution_head = HypothesisAttributionHead(
            hidden_dim=hidden_dim,
            dropout=float(attribution_config["dropout"]),
        )
        self.noisy_or = StableNoisyOR()
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

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HERADecisionModule":
        """Build the decision module from the top-level YAML config."""

        return cls(config)

    def forward(self, graph_output: HypothesisGraphOutput) -> HERADecisionOutput:
        """Forward pass using graph/evidence only; no targets are accepted."""

        graph_output.validate()
        if graph_output.refined_face is None:
            raise ValueError("HypothesisGraphOutput.refined_face is required for Phase-5 image grounding")
        if graph_output.refined_text_context is None:
            raise ValueError("HypothesisGraphOutput.refined_text_context is required for Phase-5 full-text grounding")
        atomic_logits = self.attribution_head(
            graph_output.hypothesis_features,
            graph_output.hypothesis_mask,
        )
        atomic_probabilities = torch.sigmoid(atomic_logits)
        noisy_or_output = self.noisy_or(atomic_logits)
        image_logits, image_probabilities, image_mask = self.image_grounding_head(
            hypothesis_features=graph_output.hypothesis_features,
            face_evidence=graph_output.refined_face,
        )
        text_logits, text_probabilities, text_mask = self.text_grounding_head(
            hypothesis_features=graph_output.hypothesis_features,
            text_context=graph_output.refined_text_context,
        )
        output = HERADecisionOutput(
            atomic_logits=atomic_logits,
            atomic_probabilities=atomic_probabilities,
            binary_probability=noisy_or_output.binary_probability,
            binary_logit=noisy_or_output.binary_logit,
            image_grounding_logits=image_logits,
            image_grounding_probabilities=image_probabilities,
            image_grounding_mask=image_mask,
            text_grounding_logits=text_logits,
            text_grounding_probabilities=text_probabilities,
            text_grounding_mask=text_mask,
            hypothesis_features=graph_output.hypothesis_features,
            face_source_indices=graph_output.refined_face.source_indices,
            text_word_ids=graph_output.refined_text_context.full_word_ids,
            metadata={
                "hypothesis_order": list(HYPOTHESIS_ORDER),
                "image_grounding_order": ["FS", "FA"],
                "text_grounding_order": ["TS", "TA"],
                "binary_aggregation": "noisy_or",
                "binary_shortcut": False,
            },
        )
        output.validate()
        return output

    def backbone_modules(self) -> dict[str, nn.Module]:
        """Phase 5 attaches no pretrained backbone modules directly."""

        return {}

    def new_modules(self) -> list[nn.Module]:
        """Return trainable non-backbone modules for future optimizer grouping."""

        return [self.attribution_head, self.image_grounding_head, self.text_grounding_head]
