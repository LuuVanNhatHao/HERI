from __future__ import annotations

import torch

from decision_fixtures import decision_batch, decision_config, synthetic_graph_output
from src.models.hera import HERADecisionModule


def test_decision_forward_is_target_independent() -> None:
    module = HERADecisionModule(decision_config()).eval()
    graph_output = synthetic_graph_output()
    batch_a = decision_batch()
    batch_b = decision_batch()
    batch_b["binary_target"] = 1 - batch_b["binary_target"]
    batch_b["atomic_target"] = 1 - batch_b["atomic_target"]
    batch_b["fake_image_box_target"] = batch_b["fake_image_box_target"] + 17
    batch_b["fake_text_pos"] = batch_b["fake_text_pos"] + 9
    batch_b["manipulated_face_target_index"] = torch.flip(batch_b["manipulated_face_target_index"], dims=[0])
    with torch.no_grad():
        output_a = module(graph_output)
        output_b = module(graph_output)
    assert batch_a["binary_target"].shape == batch_b["binary_target"].shape
    assert torch.equal(output_a.atomic_logits, output_b.atomic_logits)
    assert torch.equal(output_a.atomic_probabilities, output_b.atomic_probabilities)
    assert torch.equal(output_a.binary_probability, output_b.binary_probability)
    assert torch.equal(output_a.image_grounding_logits, output_b.image_grounding_logits)
    assert torch.equal(output_a.text_grounding_logits, output_b.text_grounding_logits)
