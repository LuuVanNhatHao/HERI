from __future__ import annotations

import torch

from tests.support.decision import decision_batch, decision_config, full_text_graph_output
from heri.losses.heri_components import hypothesis_text_grounding_loss
from heri.models.heri import HERADecisionModule


def test_full_text_grounding_keeps_tokens_outside_graph_budget() -> None:
    graph_output = full_text_graph_output(full_text_len=20, graph_text_nodes=5)
    module = HERADecisionModule(decision_config(token_input_dim=19)).eval()
    output = module(graph_output)
    assert graph_output.refined_text_context.features.shape[1] == 5
    assert graph_output.refined_text_context.full_features.shape[1] == 20
    assert output.text_grounding_logits.shape == (3, 2, 20)
    assert bool(output.text_grounding_mask[0, 10])
    batch = decision_batch(full_text_len=20)
    batch["atomic_target"] = torch.zeros_like(batch["atomic_target"])
    batch["atomic_target"][0, 2] = 1.0
    batch["binary_target"] = torch.tensor([1, 0, 0], dtype=torch.float32)
    batch["text_grounding_target"][:] = 0
    batch["text_grounding_target"][0, 10] = 1
    loss = hypothesis_text_grounding_loss(output, batch)
    assert torch.isfinite(loss)
    assert loss.item() > 0
