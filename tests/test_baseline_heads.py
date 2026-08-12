import torch

from src.models.heads.image_grounding import pool_candidate_patches
from src.models.outputs import PatchGrid


def test_tiny_face_uses_nearest_patch_without_target() -> None:
    patch_tokens = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    boxes = torch.tensor([[[1.0, 1.0, 2.0, 2.0]]])
    mask = torch.tensor([[True]])
    pooled = pool_candidate_patches(
        patch_tokens=patch_tokens,
        patch_grid=PatchGrid(height=2, width=2, image_height=384, image_width=384),
        candidate_boxes=boxes,
        candidate_mask=mask,
    )
    assert pooled.shape == (1, 1, 3)
    assert torch.equal(pooled[0, 0], patch_tokens[0, 0])
