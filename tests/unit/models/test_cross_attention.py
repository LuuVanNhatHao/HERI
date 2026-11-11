import torch

from heri.models.baselines.b1_cross_attention import B1CrossAttentionBaseline

from tests.support.baseline import synthetic_batch, synthetic_features


def test_b1_attention_masks_and_finite_outputs() -> None:
    model = B1CrossAttentionBaseline(
        fusion_dim=16,
        num_layers=1,
        num_heads=4,
        head_hidden_dim=16,
        dropout=0.0,
    )
    batch = synthetic_batch()
    features = synthetic_features()
    features.deberta.attention_mask[0, -2:] = False
    output = model(batch, features=features)
    assert output.atomic_logits.shape == (3, 4)
    assert output.metadata["fused_token_count"] == 18
    assert torch.isfinite(output.binary_logit).all()
