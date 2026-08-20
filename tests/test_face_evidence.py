import pytest

from evidence_fixtures import evidence_config
from src.evidence import HeterogeneousEvidenceExtractor


def test_annotation_derived_face_evidence_cannot_be_enabled() -> None:
    config = evidence_config(hidden_dim=10)
    config["face"]["enabled"] = True
    with pytest.raises(ValueError, match="removed annotation-derived MTCNN"):
        HeterogeneousEvidenceExtractor(config)
