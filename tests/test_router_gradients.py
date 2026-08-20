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
from src.routing.router import _masked_geometric_mean_confidence, _normalized_reliability_target


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


def test_padded_cross_modal_confidence_has_finite_gradients() -> None:
    visual_scores = torch.tensor([[0.8, 0.0]], requires_grad=True)
    text_scores = torch.tensor([[0.6, 0.0]], requires_grad=True)
    valid = torch.tensor([[[True, False], [False, False]]])

    confidence = _masked_geometric_mean_confidence(visual_scores, text_scores, valid)
    confidence.sum().backward()

    assert confidence.tolist() == [[[confidence[0, 0, 0].item(), 0.0], [0.0, 0.0]]]
    assert visual_scores.grad is not None and torch.isfinite(visual_scores.grad).all()
    assert text_scores.grad is not None and torch.isfinite(text_scores.grad).all()
    assert visual_scores.grad[0, 1] == 0
    assert text_scores.grad[0, 1] == 0


def test_reliability_pseudo_target_is_detached() -> None:
    scores = torch.tensor([[0.2, 0.8]], requires_grad=True)
    target = _normalized_reliability_target(scores, torch.tensor([[True, True]]))

    assert not target.requires_grad


def _hypotheses_from_routed_evidence(evidence: object) -> torch.Tensor:
    values = []
    for evidence_set in evidence.families().values():
        mask = evidence_set.mask.unsqueeze(-1).to(evidence_set.features.dtype)
        summed = (evidence_set.features * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)
        values.append(summed / count)
    pooled = torch.stack(values, dim=0).mean(dim=0)
    return pooled.unsqueeze(1).expand(-1, len(HYPOTHESIS_ORDER), -1)
