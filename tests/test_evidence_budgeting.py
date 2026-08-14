import torch

from src.evidence.budgeting import budget_features
from src.evidence.visual_forensic import forensic_patch_descriptors


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


def test_saliency_scores_are_stop_gradient_metadata() -> None:
    features = torch.randn(1, 4, 3, requires_grad=True)
    scores = torch.linalg.vector_norm(features - features.mean(dim=1, keepdim=True), dim=-1)

    budgeted = budget_features(
        features=features,
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        max_nodes=2,
        strategy="saliency_sequence",
        selection_scores=scores,
    )

    assert budgeted.selection_scores is not None
    assert not budgeted.selection_scores.requires_grad
    budgeted.features.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_forensic_descriptors_respond_to_high_frequency_artifacts() -> None:
    smooth = torch.full((1, 3, 32, 32), 0.5)
    checker = ((torch.arange(32).view(1, -1) + torch.arange(32).view(-1, 1)) % 2).float()
    checker = checker.unsqueeze(0).repeat(1, 3, 1, 1)
    fallback = torch.zeros(1, 4, 8)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)

    smooth_features, smooth_scores = forensic_patch_descriptors(
        image=smooth,
        image_content_mask=mask,
        grid_height=2,
        grid_width=2,
        fallback=fallback,
    )
    artifact_features, artifact_scores = forensic_patch_descriptors(
        image=checker,
        image_content_mask=mask,
        grid_height=2,
        grid_width=2,
        fallback=fallback,
    )

    assert artifact_scores.mean() > smooth_scores.mean()
    assert not torch.allclose(artifact_features, smooth_features)
