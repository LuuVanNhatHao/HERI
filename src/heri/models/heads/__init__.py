"""Prediction heads for conventional baselines and canonical HERA decisions."""

from .attribution import AttributionHead, HypothesisAttributionHead
from .hypothesis_grounding import (
    HypothesisConditionedImageGroundingHead,
    HypothesisConditionedTextGroundingHead,
)
from .noisy_or import StableNoisyOR

__all__ = [
    "AttributionHead",
    "HypothesisAttributionHead",
    "HypothesisConditionedImageGroundingHead",
    "HypothesisConditionedTextGroundingHead",
    "StableNoisyOR",
]
