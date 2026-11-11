from tests.support.evidence import evidence_config
from tests.support.baseline import synthetic_batch, synthetic_features
from heri.evidence import EvidenceStructuralMetadata, HeterogeneousEvidenceExtractor


def test_text_alignment_and_context_preserve_distinct_token_provenance() -> None:
    extractor = HeterogeneousEvidenceExtractor(evidence_config(hidden_dim=9))
    evidence = extractor(
        synthetic_features(), EvidenceStructuralMetadata.from_batch(synthetic_batch())
    )

    assert evidence.text_alignment.provenance["source"] == "siglip2.text_tokens"
    assert evidence.text_alignment.provenance["fake_text_pos_used"] is False
    assert evidence.text_context.provenance["source"] == "deberta.token_features"
    assert evidence.text_context.provenance["fake_text_pos_used"] is False
    assert evidence.text_alignment.source_indices.shape == evidence.text_alignment.mask.shape
    assert evidence.text_context.word_ids.shape == evidence.text_context.mask.shape
