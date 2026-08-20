from graph_fixtures import synthetic_evidence
from routing_fixtures import routing_config
from src.routing import EvidenceReliabilityRouter


def test_removed_face_family_is_absent_from_routing_contract() -> None:
    evidence = synthetic_evidence(hidden_dim=12)
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(evidence)
    assert "face" not in evidence.families()
    assert "face" not in routed.families()
    assert "face" not in routed.as_graph_evidence().families()
