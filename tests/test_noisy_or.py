from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.heads.noisy_or import StableNoisyOR, noisy_or_from_probabilities


def test_noisy_or_matches_probability_formula() -> None:
    logits = torch.tensor([[-2.0, 0.0, 1.0, 3.0], [0.5, -0.5, 2.0, -3.0]])
    output = StableNoisyOR()(logits)
    expected = 1.0 - torch.prod(1.0 - torch.sigmoid(logits), dim=1)
    assert torch.allclose(output.binary_probability, expected, atol=1e-6)
    assert torch.allclose(noisy_or_from_probabilities(torch.sigmoid(logits)), expected)


def test_noisy_or_monotonic_and_at_least_max_atomic_probability() -> None:
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    raised = probabilities.clone()
    raised[0, 2] = 0.8
    p_fake = noisy_or_from_probabilities(probabilities)
    raised_fake = noisy_or_from_probabilities(raised)
    assert raised_fake.item() >= p_fake.item()
    assert p_fake.item() >= probabilities.max().item() - 1e-6


def test_noisy_or_extreme_logits_are_finite() -> None:
    logits = torch.tensor([[-30.0, -30.0, -30.0, -30.0], [30.0, 30.0, 30.0, 30.0], [-30.0, 30.0, 0.0, 10.0]])
    output = StableNoisyOR()(logits)
    assert torch.isfinite(output.binary_probability).all()
    assert torch.isfinite(output.binary_logit).all()
    assert output.binary_probability[0].item() < 1e-10
    assert output.binary_probability[1].item() > 1.0 - 1e-6


def test_noisy_or_has_zero_trainable_parameters() -> None:
    module = StableNoisyOR()
    assert sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad) == 0


def test_noisy_or_binary_loss_backpropagates_to_all_atomic_logits() -> None:
    logits = torch.zeros(2, 4, requires_grad=True)
    output = StableNoisyOR()(logits)
    loss = F.binary_cross_entropy_with_logits(output.binary_logit, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert bool((logits.grad.abs().sum(dim=0) > 0).all())


def test_noisy_or_extreme_logits_have_finite_gradients() -> None:
    logits = torch.tensor([[-30.0, -30.0, -30.0, -30.0], [30.0, -30.0, 0.0, 10.0]], requires_grad=True)
    output = StableNoisyOR()(logits)
    loss = F.binary_cross_entropy_with_logits(output.binary_logit, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
