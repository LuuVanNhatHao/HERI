from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.support.decision import decision_batch, decision_config, synthetic_graph_output
from tests.support.graph import graph_config, synthetic_evidence
from heri.graph.reasoner import HypothesisEvidenceGraphReasoner
from heri.losses.heri_components import HERAMultitaskLoss
from heri.models.heri import HERADecisionModule


def test_generic_grounding_is_invariant_to_hypothesis_features() -> None:
    config = decision_config()
    config["model"]["grounding"]["mode"] = "generic"
    module = HERADecisionModule(config).eval()
    graph_output = synthetic_graph_output()
    changed_hypotheses = replace(
        graph_output,
        hypothesis_features=graph_output.hypothesis_features
        + 100.0 * torch.randn_like(graph_output.hypothesis_features),
    )

    original = module(graph_output)
    changed = module(changed_hypotheses)

    assert torch.equal(original.image_grounding_logits, changed.image_grounding_logits)
    assert torch.equal(original.text_grounding_logits, changed.text_grounding_logits)
    assert original.metadata["grounding_mode"] == "generic"


def test_independent_binary_head_receives_its_own_supervised_gradient() -> None:
    config = decision_config()
    config["model"]["binary"]["aggregation"] = "independent_head"
    module = HERADecisionModule(config)
    output = module(synthetic_graph_output(requires_grad=True))
    losses = HERAMultitaskLoss(config["loss"])(output, decision_batch())

    module.zero_grad(set_to_none=True)
    losses.binary.backward()

    gradients = [parameter.grad for parameter in module.binary_head.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert output.metadata["binary_aggregation"] == "independent_head"
    assert output.metadata["binary_shortcut"] is True


@pytest.mark.parametrize(
    ("graph_type", "evidence_mode", "encoder_name"),
    (
        ("gatv2", "heterogeneous", "gatv2"),
        ("hgt", "homogeneous", "homogeneous_hgt"),
    ),
)
def test_graph_ablation_runs_real_pyg_operator_on_cpu(
    graph_type: str,
    evidence_mode: str,
    encoder_name: str,
) -> None:
    pytest.importorskip("torch_geometric")
    config = graph_config(cycles=1)
    config["type"] = graph_type
    config["evidence_mode"] = evidence_mode
    reasoner = HypothesisEvidenceGraphReasoner(config).cpu().eval()

    output = reasoner(synthetic_evidence())

    assert reasoner._encoder_name == encoder_name
    assert output.hypothesis_features.shape == (3, 4, 12)
    assert torch.isfinite(output.hypothesis_features).all()
    assert output.refined_visual_semantic is not None
    assert output.refined_text_context is not None
