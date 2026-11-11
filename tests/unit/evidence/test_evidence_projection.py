import torch
from torch.nn.parameter import UninitializedParameter

from tests.support.evidence import evidence_config
from heri.evidence.cross_modal import cosine_similarity_metadata
from heri.evidence.outputs import EvidenceSet
from heri.evidence.projection import EvidenceProjectionBank


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


def test_cross_modal_similarity_uses_projected_space_despite_matching_raw_dimensions() -> None:
    mask = torch.tensor([[True]])
    sources = torch.tensor([[0]])
    visual = EvidenceSet(
        family="visual_semantic",
        features=torch.tensor([[[1.0, 0.0]]]),
        mask=mask,
        source_indices=sources,
        full_features=torch.tensor([[[0.0, 1.0]]]),
        full_mask=mask,
    )
    text = EvidenceSet(
        family="text_alignment",
        features=torch.tensor([[[-1.0, 0.0]]]),
        mask=mask,
        source_indices=sources,
        full_features=torch.tensor([[[0.0, 1.0]]]),
        full_mask=mask,
    )

    metadata = cosine_similarity_metadata(visual=visual, text=text)

    assert metadata.feature_space == "projected_evidence"
    assert metadata.similarity.item() == -1.0
