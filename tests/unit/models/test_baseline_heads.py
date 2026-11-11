import torch

from heri.models.heads.image_grounding import ImageGroundingHead
from heri.models.outputs import PatchGrid


def test_dense_image_grounding_uses_raw_patches_without_candidates() -> None:
    head = ImageGroundingHead(fusion_dim=8, hidden_dim=12, dropout=0.0)
    logits, mask, boxes, patch_boxes = head(
        patch_tokens=torch.randn(2, 4, 5),
        patch_mask=torch.ones(2, 4, dtype=torch.bool),
        patch_grid=PatchGrid(height=2, width=2, image_height=384, image_width=384),
        fused_context=torch.randn(2, 8),
    )

    assert logits.shape == (2, 4)
    assert mask.shape == (2, 4)
    assert boxes.shape == (2, 4)
    assert patch_boxes.shape == (2, 4, 4)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    assert torch.all((centers >= 0) & (centers <= 1))
    assert torch.all((sizes > 0) & (sizes <= 1))
    assert torch.all(boxes[:, 2:] >= boxes[:, :2])
