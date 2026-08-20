from __future__ import annotations

from dataclasses import replace

import torch

from graph_fixtures import synthetic_evidence
from routing_fixtures import minimal_evidence, routing_config
from src.routing import EvidenceReliabilityRouter


def test_initial_bias_starts_near_high_neutral_reliability() -> None:
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12, initial_bias=1.0)).eval()
    evidence = synthetic_evidence(hidden_dim=12)
    routed = router(evidence)
    valid_scores = torch.cat([family.reliability_scores[family.mask] for family in routed.families().values()])
    assert abs(float(valid_scores.detach().mean()) - float(torch.sigmoid(torch.tensor(1.0)))) < 0.05


def test_soft_routing_actually_modulates_feature_magnitude() -> None:
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=2, initial_bias=0.0))
    estimator = router.estimators["visual_semantic"]
    with torch.no_grad():
        estimator.hidden.weight.zero_()
        estimator.hidden.bias.zero_()
        estimator.hidden.weight[0, 0] = 1.0
        estimator.hidden.weight[0, 1] = -1.0
        estimator.output.weight.zero_()
        estimator.output.bias.fill_(-5.0)
        estimator.output.weight[0, 0] = 10.0
    routed = router(minimal_evidence(hidden_dim=2))
    scores = routed.visual_semantic.reliability_scores[0]
    magnitudes = routed.visual_semantic.routed_features.norm(dim=-1)[0]
    assert scores[0] > 0.99
    assert scores[1] < 0.01
    assert magnitudes[0] > magnitudes[1] * 50


def test_family_local_estimators_do_not_cross_depend_by_default() -> None:
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12)).eval()
    evidence = synthetic_evidence(hidden_dim=12)
    base = router(evidence).text_context.reliability_scores
    changed_visual_forensic = replace(
        evidence.visual_forensic,
        features=evidence.visual_forensic.features + torch.randn_like(evidence.visual_forensic.features) * 10.0,
    )
    changed = replace(evidence, visual_forensic=changed_visual_forensic)
    after = router(changed).text_context.reliability_scores
    assert torch.allclose(base, after)


def test_router_eval_is_deterministic() -> None:
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12)).eval()
    evidence = synthetic_evidence(hidden_dim=12)
    first = router(evidence)
    second = router(evidence)
    for family in first.families():
        assert torch.equal(first.families()[family].reliability_logits, second.families()[family].reliability_logits)


def test_enabled_router_regularization_is_finite_and_nonzero() -> None:
    config = routing_config(hidden_dim=12)
    config["routing"]["regularization"] = {
        "enabled": True,
        "weight": 0.05,
        "target_mean": 0.75,
        "min_std": 0.05,
    }
    routed = EvidenceReliabilityRouter(config)(synthetic_evidence(hidden_dim=12))
    assert routed.regularization_loss is not None
    assert torch.isfinite(routed.regularization_loss)
    assert routed.regularization_loss.item() > 0
