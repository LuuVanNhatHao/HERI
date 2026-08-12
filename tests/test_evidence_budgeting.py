import torch

from src.evidence.budgeting import budget_features


def test_sequence_balanced_budget_preserves_source_indices() -> None:
    features = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    valid_mask = torch.ones(1, 10, dtype=torch.bool)

    budgeted = budget_features(
        features=features,
        valid_mask=valid_mask,
        max_nodes=4,
        strategy="sequence_balanced",
    )

    assert budgeted.features.shape == (1, 4, 1)
    assert budgeted.mask.tolist() == [[True, True, True, True]]
    assert budgeted.source_indices.tolist() == [[0, 3, 6, 9]]
    assert budgeted.features.squeeze(-1).tolist() == [[0.0, 3.0, 6.0, 9.0]]


def test_budget_handles_zero_valid_evidence_without_nan() -> None:
    features = torch.randn(2, 5, 3)
    valid_mask = torch.zeros(2, 5, dtype=torch.bool)

    budgeted = budget_features(
        features=features,
        valid_mask=valid_mask,
        max_nodes=4,
        strategy="spatial_balanced",
    )

    assert budgeted.features.shape == (2, 0, 3)
    assert budgeted.mask.shape == (2, 0)
    assert torch.isfinite(budgeted.features).all()
