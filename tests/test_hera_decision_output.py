from __future__ import annotations

import torch

from decision_fixtures import decision_config, synthetic_graph_output
from src.graph.schema import HYPOTHESIS_ORDER
from src.models.heads.binary import BinaryHead
from src.models.heads.noisy_or import noisy_or_from_probabilities
from src.models.hera import HERADecisionModule


def test_hera_decision_output_contract_and_noisy_or_consistency() -> None:
    module = HERADecisionModule(decision_config()).eval()
    graph_output = synthetic_graph_output()
    output = module(graph_output)
    output.validate()
    assert output.atomic_logits.shape == (3, 4)
    assert output.image_grounding_logits.shape == (3, 2, 3)
    assert output.text_grounding_logits.shape == (3, 2, 5)
    expected = noisy_or_from_probabilities(output.atomic_probabilities)
    assert torch.allclose(output.binary_probability, expected, atol=1e-6)
    assert output.metadata["hypothesis_order"] == list(HYPOTHESIS_ORDER)


def test_canonical_hera_decision_module_has_no_direct_binary_head() -> None:
    module = HERADecisionModule(decision_config())
    assert not any(isinstance(child, BinaryHead) for child in module.modules())
    assert sum(parameter.numel() for parameter in module.noisy_or.parameters() if parameter.requires_grad) == 0
