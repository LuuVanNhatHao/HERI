from src.models.baselines.b0_global import B0GlobalFusionBaseline

from phase2_fixtures import synthetic_batch, synthetic_features


def test_b0_output_shapes_and_masks() -> None:
    model = B0GlobalFusionBaseline(
        fusion_dim=16,
        fusion_hidden_dim=32,
        head_hidden_dim=16,
        dropout=0.0,
    )
    batch = synthetic_batch()
    output = model(batch, features=synthetic_features())
    assert output.binary_logit.shape == (3,)
    assert output.atomic_logits.shape == (3, 4)
    assert output.image_grounding_logits.shape == (3, 4)
    assert output.image_grounding_boxes.shape == (3, 4)
    assert output.image_grounding_patch_boxes.shape == (3, 4, 4)
    assert output.text_grounding_logits.shape == batch["text_grounding_mask"].shape
    assert output.metadata["baseline_type"] == "b0_global"
