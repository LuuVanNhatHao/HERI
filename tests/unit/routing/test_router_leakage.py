from __future__ import annotations

import inspect

import torch

from tests.support.graph import synthetic_evidence
from tests.support.baseline import synthetic_batch
from tests.support.routing import routing_config
from heri.routing import EvidenceReliabilityRouter


def test_router_forward_api_does_not_accept_targets() -> None:
    signature = inspect.signature(EvidenceReliabilityRouter.forward)
    assert list(signature.parameters) == ["self", "evidence"]


def test_router_output_independent_of_changed_targets() -> None:
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12)).eval()
    evidence = synthetic_evidence(hidden_dim=12)
    batch_a = synthetic_batch()
    batch_b = synthetic_batch()
    batch_b["binary_target"] = 1 - batch_b["binary_target"]
    batch_b["atomic_target"] = 1 - batch_b["atomic_target"]
    batch_b["fake_image_box_target"] = batch_b["fake_image_box_target"] + 100
    batch_b["manipulated_face_target_index"] = torch.flip(
        batch_b["manipulated_face_target_index"], dims=[0]
    )
    batch_b["fake_text_pos"] = batch_b["fake_text_pos"] + 7
    first = router(evidence)
    second = router(evidence)
    assert batch_a["atomic_target"].shape == batch_b["atomic_target"].shape
    for family in first.families():
        assert torch.equal(
            first.families()[family].reliability_logits,
            second.families()[family].reliability_logits,
        )
        assert torch.equal(
            first.families()[family].reliability_scores,
            second.families()[family].reliability_scores,
        )
        assert torch.equal(
            first.families()[family].routed_features, second.families()[family].routed_features
        )
