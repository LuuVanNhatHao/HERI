import pytest
import torch

from heri.evidence.spatial import gather_spatial, patch_centers_and_boxes, validate_patch_grid
from heri.models.outputs import PatchGrid


def test_patch_centers_boxes_and_normalized_ranges() -> None:
    grid = PatchGrid(height=2, width=2, image_height=384, image_width=384)
    centers, boxes, norm_centers, norm_boxes = patch_centers_and_boxes(
        grid,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert centers.shape == (4, 2)
    assert boxes.shape == (4, 4)
    assert torch.allclose(centers[0], torch.tensor([96.0, 96.0]))
    assert torch.all((norm_centers >= 0) & (norm_centers <= 1))
    assert torch.all((norm_boxes >= 0) & (norm_boxes <= 1))


def test_gather_spatial_uses_zero_for_padded_indices() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source_indices = torch.tensor([[1, -1]])
    gathered = gather_spatial(values, source_indices)

    assert gathered.shape == (1, 2, 2)
    assert torch.allclose(gathered[0, 0], torch.tensor([3.0, 4.0]))
    assert torch.allclose(gathered[0, 1], torch.zeros(2))


def test_validate_patch_grid_rejects_mismatched_token_count() -> None:
    with pytest.raises(ValueError, match="grid area"):
        validate_patch_grid(
            token_count=5,
            patch_grid=PatchGrid(height=2, width=2, image_height=384, image_width=384),
            context="fixture",
        )
