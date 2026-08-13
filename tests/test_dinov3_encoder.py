from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.models.encoders.dinov3 import DINOv3Encoder


@dataclass
class DummyDINOOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor | None = None


class DummyDINOModel(torch.nn.Module):
    def __init__(self, tokens: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("tokens", tokens)

    def forward(self, *, pixel_values: torch.Tensor, return_dict: bool = True) -> DummyDINOOutput:
        batch = pixel_values.shape[0]
        return DummyDINOOutput(last_hidden_state=self.tokens[:batch].to(pixel_values.device))


class ChunkedDINOModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.blocks = [
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
        ]
        self.encoder.layer_chunks = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(self.blocks[:2]),
                torch.nn.ModuleList(self.blocks[2:]),
            ]
        )


def test_dinov3_strips_prefix_register_tokens_for_square_patch_grid() -> None:
    tokens = torch.arange(581 * 4, dtype=torch.float32).reshape(1, 581, 4)
    encoder = DINOv3Encoder(
        model=DummyDINOModel(tokens),
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        model_name="dummy-dinov3",
    )

    output = encoder(image=torch.zeros(1, 3, 384, 384))

    assert output.patch_tokens.shape == (1, 576, 4)
    assert output.patch_grid.height == 24
    assert output.patch_grid.width == 24
    assert output.patch_special_token_count == 5
    assert torch.equal(output.patch_tokens, tokens[:, 5:, :])
    assert torch.equal(output.global_feature, tokens[:, 0, :])


def test_dinov3_patch_grid_failure_names_token_count() -> None:
    tokens = torch.zeros(1, 600, 4)
    encoder = DINOv3Encoder(
        model=DummyDINOModel(tokens),
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        model_name="dummy-dinov3",
    )

    with pytest.raises(ValueError, match="600 tokens"):
        encoder(image=torch.zeros(1, 3, 384, 384))


def test_dinov3_partial_unfreeze_supports_chunked_encoder_blocks() -> None:
    model = ChunkedDINOModel()
    encoder = DINOv3Encoder(
        model=model,
        image_mean=(0.0, 0.0, 0.0),
        image_std=(1.0, 1.0, 1.0),
        model_name="dummy-dinov3-chunked",
    )
    encoder.freeze_all()

    changed = encoder.unfreeze_last_blocks(2)

    assert changed == sum(parameter.numel() for block in model.blocks[-2:] for parameter in block.parameters())
    assert all(not parameter.requires_grad for block in model.blocks[:-2] for parameter in block.parameters())
    assert all(parameter.requires_grad for block in model.blocks[-2:] for parameter in block.parameters())
