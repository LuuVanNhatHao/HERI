from __future__ import annotations

from dataclasses import replace

from decision_fixtures import decision_config, full_text_graph_output
from routing_fixtures import routing_config
from src.evidence.outputs import CrossModalMetadata, HeterogeneousEvidenceBatch
from src.models.hera import HERADecisionModule
from src.routing import EvidenceReliabilityRouter


def test_router_preserves_full_text_store_for_phase5_grounding() -> None:
    evidence = full_text_graph_output(full_text_len=20, graph_text_nodes=5)
    original = evidence.refined_text_context
    batch = HeterogeneousEvidenceBatch(
        visual_semantic=evidence.refined_visual_semantic,
        visual_forensic=evidence.refined_visual_forensic,
        face=evidence.refined_face,
        text_alignment=evidence.refined_text_alignment,
        text_context=original,
        cross_modal_metadata=CrossModalMetadata(method="disabled"),
    )
    routed = EvidenceReliabilityRouter(routing_config(hidden_dim=12))(batch)
    routed_text = routed.as_graph_evidence().text_context
    assert routed_text.full_features is original.full_features
    assert routed_text.full_mask is original.full_mask
    assert routed_text.full_word_ids is original.full_word_ids
    routed_graph_output = replace(evidence, refined_text_context=routed_text)
    output = HERADecisionModule(decision_config()).eval()(routed_graph_output)
    assert output.text_grounding_logits.shape[-1] == 20
    assert bool(output.text_grounding_mask[0, 10])
