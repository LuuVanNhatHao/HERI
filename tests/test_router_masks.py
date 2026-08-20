from __future__ import annotations

import torch

from graph_fixtures import synthetic_evidence
from routing_fixtures import routing_config
from src.routing import EvidenceReliabilityRouter


def test_removed_face_family_stays_empty_and_inactive() -> None:
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(synthetic_evidence(hidden_dim=12))
    face = routed.face
    assert face.mask.tolist() == [[], [], []]
    assert torch.all(face.reliability_scores[~face.mask] == 0)
    assert torch.all(face.routed_features[~face.mask] == 0)


def test_hard_routing_uses_yaml_threshold_without_activating_padded_nodes() -> None:
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12, mode="hard"))
    routed = router(synthetic_evidence(hidden_dim=12))
    for routed_set in routed.families().values():
        assert routed_set.effective_mask is not None
        assert not bool((routed_set.effective_mask & ~routed_set.mask).any())
    routed.as_graph_evidence().validate()
