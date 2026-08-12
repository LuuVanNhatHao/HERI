from __future__ import annotations

import torch

from decision_fixtures import decision_batch, decision_config
from graph_fixtures import graph_config, synthetic_evidence
from routing_fixtures import routing_config
from src.graph.builder import HypothesisEvidenceGraphBuilder
from src.graph.outputs import HypothesisGraphOutput
from src.graph.schema import HYPOTHESIS_ORDER
from src.graph.validation import graph_statistics
from src.losses.hera_components import HERAMultitaskLoss
from src.models.hera import HERADecisionModule
from src.routing import EvidenceReliabilityRouter


def test_downstream_phase5_loss_backpropagates_to_router_parameters() -> None:
    router = EvidenceReliabilityRouter(routing_config(hidden_dim=12))
    routed_evidence = router(synthetic_evidence(hidden_dim=12)).as_graph_evidence()
    hypothesis_features = _hypotheses_from_routed_evidence(routed_evidence)
    graph = HypothesisEvidenceGraphBuilder(graph_config(hidden_dim=12)).build(routed_evidence, hypothesis_features)
    graph_output = HypothesisGraphOutput(
        hypothesis_features=hypothesis_features,
        hypothesis_mask=torch.ones(3, len(HYPOTHESIS_ORDER), dtype=torch.bool),
        graph_batch=graph,
        graph_statistics=graph_statistics(graph),
        updated_node_features=graph.x_dict,
        refined_visual_semantic=routed_evidence.visual_semantic,
        refined_visual_forensic=routed_evidence.visual_forensic,
        refined_face=routed_evidence.face,
        refined_text_alignment=routed_evidence.text_alignment,
        refined_text_context=routed_evidence.text_context,
    )
    output = HERADecisionModule(decision_config())(graph_output)
    losses = HERAMultitaskLoss(decision_config()["loss"])(output, decision_batch())
    (losses.attribution + losses.binary).backward()
    grads = [parameter.grad for parameter in router.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(float(grad.abs().sum()) > 0 for grad in grads)


def _hypotheses_from_routed_evidence(evidence: object) -> torch.Tensor:
    values = []
    for evidence_set in evidence.families().values():
        mask = evidence_set.mask.unsqueeze(-1).to(evidence_set.features.dtype)
        summed = (evidence_set.features * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)
        values.append(summed / count)
    pooled = torch.stack(values, dim=0).mean(dim=0)
    return pooled.unsqueeze(1).expand(-1, len(HYPOTHESIS_ORDER), -1)
