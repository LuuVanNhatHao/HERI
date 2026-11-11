from types import SimpleNamespace

import pytest
import torch

from heri.models.encoders.deberta import configure_reference_compile


class _ConfigurableEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(reference_compile=None)


def test_reference_compile_is_disabled_before_staged_training() -> None:
    model = _ConfigurableEncoder()

    actual = configure_reference_compile(model, requested=False)

    assert actual is False
    assert model.config.reference_compile is False


def test_reference_compile_setting_fails_if_backbone_cannot_honor_it() -> None:
    with pytest.raises(RuntimeError, match="does not expose"):
        configure_reference_compile(torch.nn.Linear(2, 2), requested=False)
