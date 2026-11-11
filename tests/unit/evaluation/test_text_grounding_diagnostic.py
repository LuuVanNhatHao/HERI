from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from heri.evaluation.evaluator import (
    PredictionBundle,
    _accumulate_text_counts,
    text_grounding_metrics,
)
from heri.evaluation.hypothesis_grounding import (
    route_text_grounding_probabilities,
)


def _output() -> SimpleNamespace:
    atomic = torch.tensor([[0.01, 0.01], [0.90, 0.90]])
    heads = torch.tensor(
        [
            [[0.90, 0.10], [0.70, 0.10]],
            [[0.90, 0.10], [0.70, 0.10]],
        ]
    )
    return SimpleNamespace(
        atomic_probabilities=atomic,
        text_grounding_probabilities=heads,
        text_grounding_logits=torch.logit(heads),
        text_grounding_mask=torch.ones(2, 2, dtype=torch.bool),
        metadata={
            "text_hypothesis_indices": [0, 1],
            "grounding_routing": "soft",
            "grounding_routing_temperature": 1.0,
        },
    )


def test_presence_aware_aggregations_retain_absolute_manipulation_probability() -> None:
    output = _output()
    relative, weights = route_text_grounding_probabilities(output, aggregation="relative_soft")
    gated, _ = route_text_grounding_probabilities(output, aggregation="presence_gated")
    noisy_or, _ = route_text_grounding_probabilities(output, aggregation="token_noisy_or")

    # Relative routing is deliberately scale invariant: this is the released
    # behavior whose SAMM absence failure the diagnostic must reproduce.
    assert torch.allclose(weights[0], weights[1])
    assert torch.allclose(relative[0], relative[1])
    assert relative[0, 0] > 0.5
    assert gated[0, 0] < 0.5 < gated[1, 0]
    assert noisy_or[0, 0] < 0.5 < noisy_or[1, 0]
    assert torch.all((gated >= 0.0) & (gated <= 1.0))
    assert torch.all((noisy_or >= 0.0) & (noisy_or <= 1.0))


def test_absence_diagnostic_counts_orig_false_positive_words() -> None:
    output = _output()
    batch = {
        "text_grounding_target": torch.tensor([[0, 0], [1, 0]]),
        "text_word_ids": torch.tensor([[0, 1], [0, 1]]),
        "has_text_manipulation": torch.tensor([False, True]),
        "fake_text_pos": torch.tensor([[-1, -1], [0, -1]]),
        "words": [["real", "caption"], ["changed", "caption"]],
    }
    relative = PredictionBundle(text_grounding_aggregation="relative_soft")
    gated = PredictionBundle(text_grounding_aggregation="presence_gated")
    _accumulate_text_counts(relative, output, batch, text_grounding_aggregation="relative_soft")
    _accumulate_text_counts(
        gated,
        output,
        batch,
        text_grounding_aggregation="presence_gated",
        store_details=False,
    )

    relative_metrics = text_grounding_metrics(relative)
    gated_metrics = text_grounding_metrics(gated)
    assert relative_metrics["text_absent_sample_count"] == 1
    assert relative_metrics["text_absent_word_predicted_positive_count"] == 1
    assert relative_metrics["text_absent_sample_false_positive_rate"] == 1.0
    assert gated_metrics["text_absent_word_predicted_positive_count"] == 0
    assert gated_metrics["text_absent_sample_false_positive_rate"] == 0.0
    assert relative_metrics["text_f1_positive_only"] == pytest.approx(1.0)
    assert gated_metrics["text_f1_positive_only"] == pytest.approx(1.0)


def test_unknown_text_grounding_aggregation_is_rejected() -> None:
    with pytest.raises(ValueError, match="text grounding aggregation"):
        route_text_grounding_probabilities(_output(), aggregation="oracle")
