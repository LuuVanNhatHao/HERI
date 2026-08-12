from __future__ import annotations

from dataclasses import replace

import torch

from decision_fixtures import full_text_graph_output, synthetic_graph_output
from src.evidence.outputs import PAD_WORD_ID
from src.models.heads.hypothesis_grounding import HypothesisConditionedTextGroundingHead


def test_text_grounding_scores_ts_and_ta_full_tokens() -> None:
    graph_output = synthetic_graph_output()
    head = HypothesisConditionedTextGroundingHead(hidden_dim=12, token_input_dim=19, dropout=0.0)
    logits, probabilities, mask = head(
        hypothesis_features=graph_output.hypothesis_features,
        text_context=graph_output.refined_text_context,
    )
    assert logits.shape == (3, 2, 5)
    assert probabilities.shape == logits.shape
    assert mask.shape == (3, 5)
    assert torch.all(probabilities.masked_select(~mask.unsqueeze(1)) == 0)


def test_text_grounding_masks_special_padding_and_invalid_word_ids() -> None:
    graph_output = full_text_graph_output(full_text_len=6, graph_text_nodes=3)
    text_context = graph_output.refined_text_context
    full_mask = text_context.full_mask.clone()
    full_word_ids = text_context.full_word_ids.clone()
    full_mask[0, 4] = False
    full_word_ids[0, 5] = PAD_WORD_ID
    text_context = replace(text_context, full_mask=full_mask, full_word_ids=full_word_ids)
    head = HypothesisConditionedTextGroundingHead(hidden_dim=12, token_input_dim=19, dropout=0.0)
    _, probabilities, mask = head(hypothesis_features=graph_output.hypothesis_features, text_context=text_context)
    assert not bool(mask[0, 4])
    assert not bool(mask[0, 5])
    assert torch.equal(probabilities[0, :, 4], torch.zeros(2))
    assert torch.equal(probabilities[0, :, 5], torch.zeros(2))


def test_text_grounding_uses_corresponding_hypothesis_state() -> None:
    graph_output = synthetic_graph_output()
    head = HypothesisConditionedTextGroundingHead(hidden_dim=12, token_input_dim=19, dropout=0.0).eval()
    base_logits, _, _ = head(
        hypothesis_features=graph_output.hypothesis_features,
        text_context=graph_output.refined_text_context,
    )
    altered = graph_output.hypothesis_features.clone()
    altered[:, 2, :] += torch.randn_like(altered[:, 2, :]) * 3.0
    changed_logits, _, _ = head(hypothesis_features=altered, text_context=graph_output.refined_text_context)
    assert not torch.allclose(base_logits[:, 0, :], changed_logits[:, 0, :])
    assert torch.allclose(base_logits[:, 1, :], changed_logits[:, 1, :])
