from __future__ import annotations

from tests.support.graph import synthetic_evidence
from tests.support.routing import routing_config
from heri.routing import EvidenceReliabilityRouter


def test_removed_face_family_is_not_registered() -> None:
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(
        synthetic_evidence(hidden_dim=12)
    )
    assert "face" not in routed.families()


def test_hard_routing_uses_yaml_threshold_without_activating_padded_nodes() -> None:
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12, mode="hard"))
    routed = router(synthetic_evidence(hidden_dim=12))
    for routed_set in routed.families().values():
        assert routed_set.effective_mask is not None
        assert not bool((routed_set.effective_mask & ~routed_set.mask).any())
    routed.as_graph_evidence().validate()
