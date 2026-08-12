import torch

from src.losses.baseline_losses import BaselineMultitaskLoss
from src.models.baselines.b0_global import B0GlobalFusionBaseline

from phase2_fixtures import synthetic_batch, synthetic_features


def test_baseline_losses_and_zero_applicability_masks() -> None:
    model = B0GlobalFusionBaseline(
        fusion_dim=16,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        dropout=0.0,
    )
    batch = synthetic_batch()
    output = model(batch, features=synthetic_features())
    loss_fn = BaselineMultitaskLoss(
        {"binary": 1.0, "attribution": 1.0, "image_grounding": 1.0, "text_grounding": 1.0}
    )
    losses = loss_fn(output, batch)
    assert torch.isfinite(losses.total)

    batch["has_image_manipulation"] = torch.zeros_like(batch["has_image_manipulation"])
    batch["has_text_manipulation"] = torch.zeros_like(batch["has_text_manipulation"])
    masked = loss_fn(output, batch)
    assert masked.image_grounding.item() == 0.0
    assert masked.text_grounding.item() == 0.0
