from __future__ import annotations

import torch

from hera_phase7_fixtures import FixtureGraphReasoner, hera_test_config, perturb_forbidden_targets
from phase2_fixtures import synthetic_batch, synthetic_features
from src.losses.hera_components import HERAMultitaskLoss
from src.models.heads.noisy_or import noisy_or_from_probabilities
from src.models.hera import HERADGM, HERADGMOutput
from src.training.optimizer import apply_training_stage


def _hera_model(routing_enabled: bool = True) -> HERADGM:
    config = hera_test_config(routing_enabled=routing_enabled)
    return HERADGM(
        config,
        graph_reasoner=FixtureGraphReasoner(hidden_dim=12),
        load_backbones=False,
    ).eval()


def test_hera_dgm_composes_phase3_to_phase6_modules_and_outputs_contract() -> None:
    model = _hera_model()
    output = model(synthetic_batch(), features=synthetic_features())

    assert isinstance(output, HERADGMOutput)
    assert output.atomic_logits.shape == (3, 4)
    assert output.binary_probability.shape == (3,)
    assert output.image_grounding_logits.ndim == 3
    assert output.text_grounding_logits.shape[-1] == 5
    assert output.metadata["pipeline"][-1] == "StableNoisyOR"
    assert output.metadata["binary_shortcut"] is False
    output.validate()


def test_hera_forward_is_independent_of_forbidden_target_fields() -> None:
    model = _hera_model()
    batch = synthetic_batch()
    changed_targets = perturb_forbidden_targets(batch)

    first = model(batch, features=synthetic_features())
    second = model(changed_targets, features=synthetic_features())

    for name in (
        "atomic_logits",
        "binary_probability",
        "image_grounding_logits",
        "text_grounding_logits",
        "hypothesis_features",
    ):
        assert torch.allclose(getattr(first, name), getattr(second, name), atol=1e-6)


def test_hera_has_no_independent_binary_shortcut_and_uses_noisy_or() -> None:
    model = _hera_model()
    output = model(synthetic_batch(), features=synthetic_features())
    expected_fake = noisy_or_from_probabilities(output.atomic_probabilities)

    assert model.count_independent_binary_shortcuts() == 0
    assert torch.allclose(output.binary_probability, expected_fake, atol=1e-6)
    assert not any("binary_head" in name for name, _ in model.named_modules())


def test_routing_enabled_and_bypass_paths_share_output_contract() -> None:
    batch = synthetic_batch()
    features = synthetic_features()
    enabled = _hera_model(routing_enabled=True)(batch, features=features)
    bypass = _hera_model(routing_enabled=False)(batch, features=features)

    assert enabled.metadata["router_batch_summary"]["router"]["enabled"] is True
    assert bypass.metadata["router_batch_summary"]["router"]["enabled"] is False
    assert enabled.atomic_logits.shape == bypass.atomic_logits.shape
    assert bypass.metadata["router_family_mean_reliability"]["face"][0] == 0.0


def test_router_parameters_are_frozen_and_excluded_when_bypassed() -> None:
    model = _hera_model(routing_enabled=False)
    apply_training_stage(model, hera_test_config(routing_enabled=False), epoch=1)

    assert all(not parameter.requires_grad for parameter in model.router.parameters())
    assert model.router not in model.new_modules()


def test_full_hera_loss_includes_evidence_term_and_mixed_sample_is_finite() -> None:
    config = hera_test_config()
    config["loss"]["lambda_ev"] = 0.25
    model = _hera_model()
    output = model(synthetic_batch(), features=synthetic_features())
    losses = HERAMultitaskLoss(config["loss"])(output, synthetic_batch())

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.evidence)
    assert "evidence" in losses.as_dict()
