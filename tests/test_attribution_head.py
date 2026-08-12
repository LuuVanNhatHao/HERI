from __future__ import annotations

import torch
import torch.nn.functional as F

from src.graph.schema import HYPOTHESIS_ORDER
from src.models.heads.attribution import HypothesisAttributionHead


def test_hypothesis_attribution_preserves_order_and_shape() -> None:
    head = HypothesisAttributionHead(hidden_dim=12, dropout=0.0)
    features = torch.randn(2, len(HYPOTHESIS_ORDER), 12)
    logits = head(features, torch.ones(2, 4, dtype=torch.bool))
    assert logits.shape == (2, 4)
    assert head.order == HYPOTHESIS_ORDER


def test_hypothesis_attribution_is_multilabel_sigmoid() -> None:
    head = HypothesisAttributionHead(hidden_dim=12, dropout=0.0)
    features = torch.randn(2, 4, 12)
    logits = head(features)
    target = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    assert torch.isfinite(loss)
    probabilities = torch.sigmoid(logits)
    assert probabilities.shape == (2, 4)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
