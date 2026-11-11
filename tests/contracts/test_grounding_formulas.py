"""Small numerical examples for HERI Eq.18--26,32--33; no model downloads."""

from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from heri.evaluation.hypothesis_grounding import (
    route_image_grounding_boxes,
    route_text_grounding_probabilities,
)
from heri.evaluation.evaluator import _clip_and_restore_box
from heri.losses.box_grounding import dense_box_grounding_loss
from heri.losses.heri_components import hypothesis_image_grounding_loss
from heri.metrics.consistency import evidence_decision_metrics
from heri.metrics.verified_support import verified_support_units
from heri.models.heads.hypothesis_grounding import (
    _masked_softmax,
    _normalized_center_size_box,
)


def test_eq19_invalid_scores_cannot_drain_attention_and_empty_rows_are_zero():
    logits = torch.tensor([[0.0, 100.0], [100.0, 100.0]], dtype=torch.bfloat16)
    weights = _masked_softmax(logits, torch.tensor([[True, False], [False, False]]))
    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert torch.isfinite(weights).all()


def test_eq20_sigmoid_center_size_is_not_anchor_in_available_room():
    parameters = torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, -2.0, 2.0, 2.0]])
    boxes = _normalized_center_size_box(parameters)
    torch.testing.assert_close(boxes[0], torch.tensor([0.25, 0.25, 0.75, 0.75]))
    # Off-canvas extents are clipped only during inference, after training.
    assert boxes[1, 2] > 1 and boxes[1, 1] < 0


def test_eq21_and_22_use_relative_text_weights_but_only_active_image_heads():
    output = SimpleNamespace(
        atomic_probabilities=torch.tensor([[0.3, 0.3], [0.8, 0.4]]),
        image_grounding_boxes=torch.tensor([[[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.9, 0.9]]] * 2),
        text_grounding_probabilities=torch.tensor([[[0.2, 0.8], [0.8, 0.2]]] * 2),
        metadata={
            "image_hypothesis_indices": [0, 1],
            "text_hypothesis_indices": [0, 1],
        },
    )
    image, weights = route_image_grounding_boxes(output)
    torch.testing.assert_close(image[0], torch.zeros(4))
    torch.testing.assert_close(weights[0], torch.zeros(2))
    torch.testing.assert_close(image[1], torch.tensor([0.1, 0.1, 0.2, 0.2]))
    text, _ = route_text_grounding_probabilities(output)
    torch.testing.assert_close(text[0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(text[1], torch.tensor([0.4, 0.6]))


def test_eq22_removes_padding_and_restores_original_pixels():
    meta = {
        "pad_left": 0,
        "pad_top": 2,
        "resized_width": 8,
        "resized_height": 4,
        "scale": 0.5,
        "original_width": 16,
        "original_height": 8,
    }
    clipped, original = _clip_and_restore_box(torch.tensor([-0.2, 0.0, 1.2, 1.0]), meta, 8, 8)
    torch.testing.assert_close(clipped, torch.tensor([0.0, 0.25, 1.0, 0.75]))
    torch.testing.assert_close(original, torch.tensor([0.0, 0.0, 16.0, 8.0]))


def test_eq26_averages_valid_pairs_rather_than_means_of_heads():
    target = torch.tensor([0.25, 0.25, 0.75, 0.75])
    bad = torch.tensor([0.0, 0.0, 0.1, 0.1])
    output = SimpleNamespace(
        image_grounding_logits=torch.zeros(2, 2, 1),
        image_grounding_mask=torch.ones(2, 1, dtype=torch.bool),
        image_grounding_boxes=torch.stack((torch.stack((target, bad)), torch.stack((target, bad)))),
        image_grounding_patch_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0]]] * 2),
        metadata={"image_hypothesis_indices": [0, 1]},
    )
    batch = {
        "image": torch.zeros(2, 3, 8, 8),
        "fake_image_box_target": torch.tensor([[2.0, 2.0, 6.0, 6.0]] * 2),
        "atomic_target": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
    }
    loss = hypothesis_image_grounding_loss(output, batch, heatmap_weight=0.0)
    bad_loss = dense_box_grounding_loss(
        predicted_boxes=bad[None],
        patch_logits=torch.zeros(1, 1),
        patch_mask=torch.ones(1, 1, dtype=torch.bool),
        patch_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
        target_boxes=target[None],
        valid_rows=torch.tensor([True]),
        heatmap_weight=0.0,
    )
    torch.testing.assert_close(loss, bad_loss / 3)


def test_eq26_patch_means_are_nested_inside_equal_pair_averaging():
    targets = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 2)
    value = dense_box_grounding_loss(
        predicted_boxes=targets,
        target_boxes=targets,
        patch_logits=torch.tensor([[0.0, 0.0], [math.log(3), 100.0]]),
        patch_mask=torch.tensor([[True, True], [True, False]]),
        patch_boxes=torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]] * 2),
        valid_rows=torch.tensor([True, True]),
        l1_weight=0.0,
        giou_weight=0.0,
        heatmap_weight=1.0,
    )
    assert value.item() == pytest.approx((math.log(2) + math.log(4 / 3)) / 2)


def test_eq32_is_unweighted_and_unsupported_means_support_below_half():
    result = evidence_decision_metrics(
        np.array([[0.9, 0.5]]), np.array([[0.7, 0.5]]), np.array([0.5, 0.5])
    )
    assert result["edc"] == pytest.approx((0.7 / (0.9 + 1e-8) + 0.5 / (0.5 + 1e-8)) / 2)
    assert result["udr"] == 0.0


def test_eq33_requires_top_support_inside_annotation_and_true_positive_hypothesis():
    hits, eligible = verified_support_units(
        atomic_probabilities=torch.tensor([[0.9, 0.9], [0.9, 0.2]]),
        atomic_targets=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        grounding_logits=torch.tensor(
            [
                [[2.0, 4.0, 100.0], [2.0, 1.0, 100.0]],
                [[4.0, 2.0, 100.0], [4.0, 2.0, 100.0]],
            ]
        ),
        grounding_mask=torch.tensor([[True, True, False]] * 2),
        localization_targets=torch.tensor([[True, False, False]] * 2),
        hypothesis_indices=(0, 1),
        annotation_available=torch.tensor([True, True]),
    )
    assert eligible.tolist() == [[True, False], [True, False]]
    assert hits.tolist() == [[False, False], [True, False]]
