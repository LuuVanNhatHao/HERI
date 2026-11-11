"""Loss functions for HERA-DGM phases."""

from .heri_components import (
    HERALossResult,
    HERAMultitaskLoss,
    evidence_sufficiency_loss,
    hypothesis_image_grounding_loss,
    hypothesis_text_grounding_loss,
    image_grounding_applicability,
    text_grounding_applicability,
)

__all__ = [
    "HERALossResult",
    "HERAMultitaskLoss",
    "evidence_sufficiency_loss",
    "hypothesis_image_grounding_loss",
    "hypothesis_text_grounding_loss",
    "image_grounding_applicability",
    "text_grounding_applicability",
]
