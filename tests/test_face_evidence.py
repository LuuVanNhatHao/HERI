from evidence_fixtures import evidence_config
from graph_fixtures import synthetic_evidence


def test_annotation_derived_face_evidence_is_absent_from_final_contract() -> None:
    config = evidence_config(hidden_dim=10)
    assert "face" not in config
    assert "face" not in synthetic_evidence(hidden_dim=10).families()
