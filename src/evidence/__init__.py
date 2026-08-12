"""Phase-3 heterogeneous evidence extraction layer."""

from .extractor import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor
from .outputs import CrossModalMetadata, EvidenceSet, HeterogeneousEvidenceBatch

__all__ = [
    "CrossModalMetadata",
    "EvidenceSet",
    "EvidenceStructuralMetadata",
    "HeterogeneousEvidenceBatch",
    "HeterogeneousEvidenceExtractor",
]
