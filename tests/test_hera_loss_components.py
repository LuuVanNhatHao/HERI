from __future__ import annotations

from dataclasses import replace

import torch

from decision_fixtures import decision_batch, decision_config, synthetic_graph_output
from src.losses.hera_components import (
    HERAMultitaskLoss,
    hypothesis_image_grounding_loss,
    hypothesis_text_grounding_loss,
    image_grounding_applicability,
    text_grounding_applicability,
    target_consistent_noisy_or_loss,
)
from src.models.hera import HERADecisionModule


def test_hera_loss_components_are_finite_and_backward_reaches_attribution() -> None:
    module = HERADecisionModule(decision_config())
    graph_output = synthetic_graph_output(requires_grad=True)
    output = module(graph_output)
    batch = decision_batch()
    losses = HERAMultitaskLoss(decision_config()["loss"])(output, batch)
    assert torch.isfinite(losses.total)
    module.zero_grad(set_to_none=True)
    losses.binary.backward(retain_graph=True)
    grads = [parameter.grad for parameter in module.attribution_head.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_zero_grounding_applicability_is_safe_zero() -> None:
    module = HERADecisionModule(decision_config()).eval()
    output = module(synthetic_graph_output())
    batch = decision_batch()
    batch["atomic_target"] = torch.zeros_like(batch["atomic_target"])
    image_loss = hypothesis_image_grounding_loss(output, batch)
    text_loss = hypothesis_text_grounding_loss(output, batch)
    assert torch.isfinite(image_loss)
    assert torch.isfinite(text_loss)
    assert image_loss.item() == 0.0
    assert text_loss.item() == 0.0


def test_mixed_fs_ts_sample_supervises_only_relevant_grounding_heads() -> None:
    module = HERADecisionModule(decision_config()).eval()
    output = module(synthetic_graph_output())
    batch = decision_batch()
    batch["atomic_target"] = torch.tensor(
        [[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.float32,
    )
    batch["binary_target"] = torch.tensor([1, 0, 0], dtype=torch.float32)
    batch["manipulated_face_target_index"] = torch.tensor([0, -1, -1], dtype=torch.long)
    batch["text_grounding_target"][:] = 0
    batch["text_grounding_target"][0, 2] = 1
    assert image_grounding_applicability(batch["atomic_target"]).tolist()[0] == [True, False]
    assert text_grounding_applicability(batch["atomic_target"]).tolist()[0] == [True, False]
    base_image = hypothesis_image_grounding_loss(output, batch)
    base_text = hypothesis_text_grounding_loss(output, batch)
    altered_image = replace(output, image_grounding_logits=output.image_grounding_logits.clone())
    altered_image.image_grounding_logits[:, 1, :] = 999.0
    altered_text = replace(output, text_grounding_logits=output.text_grounding_logits.clone())
    altered_text.text_grounding_logits[:, 1, :] = 999.0
    assert torch.allclose(base_image, hypothesis_image_grounding_loss(altered_image, batch))
    assert torch.allclose(base_text, hypothesis_text_grounding_loss(altered_text, batch))


def test_fs_fa_ts_ta_single_applicability_losses_are_finite() -> None:
    module = HERADecisionModule(decision_config()).eval()
    output = module(synthetic_graph_output())
    for atomic_index in range(4):
        batch = decision_batch()
        batch["atomic_target"] = torch.zeros_like(batch["atomic_target"])
        batch["atomic_target"][0, atomic_index] = 1.0
        batch["binary_target"] = torch.tensor([1, 0, 0], dtype=torch.float32)
        batch["manipulated_face_target_index"] = torch.tensor([0, -1, -1], dtype=torch.long)
        batch["text_grounding_target"][:] = 0
        batch["text_grounding_target"][0, 0] = 1
        losses = HERAMultitaskLoss(decision_config()["loss"])(output, batch)
        assert torch.isfinite(losses.total)


def test_binary_loss_does_not_push_absent_hypotheses_up_on_fake_rows() -> None:
    logits = torch.tensor([[1.0, -0.5, 0.25, -1.0]], requires_grad=True)
    atomic_target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    binary_target = torch.tensor([1.0])

    loss = target_consistent_noisy_or_loss(logits, atomic_target, binary_target)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0] < 0
    assert torch.equal(logits.grad[0, 1:], torch.zeros(3))
