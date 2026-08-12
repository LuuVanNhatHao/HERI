import torch
from torch.nn.parameter import UninitializedParameter

from evidence_fixtures import evidence_config
from src.evidence.projection import EvidenceProjectionBank


def test_family_specific_projectors_are_distinct_and_materialize() -> None:
    bank = EvidenceProjectionBank(evidence_config(hidden_dim=8))
    assert bank["visual_semantic"] is not bank["visual_forensic"]
    assert bank["text_alignment"] is not bank["text_context"]
    assert any(isinstance(parameter, UninitializedParameter) for parameter in bank.parameters())

    output = bank["visual_semantic"](torch.randn(2, 3, 5))

    assert output.shape == (2, 3, 8)
    assert not any(
        isinstance(parameter, UninitializedParameter)
        for parameter in bank["visual_semantic"].parameters()
    )
