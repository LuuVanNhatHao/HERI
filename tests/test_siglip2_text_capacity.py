from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from src.models.encoders.siglip2 import SigLIP2Encoder


class _AdvertisedCapacityModel(nn.Module):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=capacity)
        )


def _encoder(max_length: int) -> SigLIP2Encoder:
    return SigLIP2Encoder(
        model=_AdvertisedCapacityModel(64),
        tokenizer=object(),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        max_length=max_length,
        model_name="test/siglip2",
    )


def test_siglip2_accepts_text_length_at_pretrained_capacity() -> None:
    assert _encoder(64).max_length == 64


def test_siglip2_rejects_text_length_above_pretrained_capacity() -> None:
    with pytest.raises(
        ValueError,
        match=r"text_max_length=128.*max_position_embeddings=64",
    ):
        _encoder(128)
