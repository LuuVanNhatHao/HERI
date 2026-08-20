from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from hera_phase7_fixtures import FixtureGraphReasoner, hera_test_config
from phase2_fixtures import synthetic_batch, synthetic_features
from src.benchmarks.contracts import SignalContract
from src.benchmarks.loss import BinaryBenchmarkLoss
from src.benchmarks.metrics import (
    BinaryPredictionBundle,
    auxiliary_metrics,
    binary_metrics_and_report,
    tune_binary_threshold,
)
from src.benchmarks.model import BinaryBenchmarkHERA


def _binary_config(*, subtype: bool = False, text_label: bool = False) -> dict:
    config = hera_test_config(hidden_dim=12)
    config["task"] = {
        "type": "binary_multimodal",
        "signals": {
            "binary": True,
            "attribution": False,
            "image_grounding": False,
            "text_grounding": False,
            "subtype": subtype,
            "text_label": text_label,
        },
    }
    config["model"]["model_type"] = "hera_binary_benchmark"
    config["model"]["binary"]["aggregation"] = "independent_head"
    config["model"]["binary"]["dropout"] = 0.0
    config["model"]["latent_slots"] = {
        "count": 4,
        "roles": ["visual_slot_1", "visual_slot_2", "text_slot_1", "text_slot_2"],
    }
    config["loss"] = {
        "lambda_bin": 1.0,
        "lambda_align": 0.1,
        "lambda_router": 0.05,
        "lambda_subtype": 0.2 if subtype else 0.0,
        "lambda_text_label": 0.2 if text_label else 0.0,
    }
    return config


def _batch() -> dict:
    batch = synthetic_batch()
    batch["binary_available"] = torch.ones(3, dtype=torch.bool)
    batch["subtype_target"] = torch.tensor([0, 1, 2])
    batch["subtype_available"] = torch.ones(3, dtype=torch.bool)
    batch["text_label_target"] = torch.tensor([0.0, 1.0, 1.0])
    batch["text_label_available"] = torch.ones(3, dtype=torch.bool)
    return batch


def test_binary_hera_uses_fusion_reasoning_without_dgm4_decision_heads() -> None:
    model = BinaryBenchmarkHERA(
        _binary_config(),
        graph_reasoner=FixtureGraphReasoner(hidden_dim=12),
        load_backbones=False,
    ).eval()
    output = model(_batch(), features=synthetic_features())

    assert output.binary_logit.shape == (3,)
    assert output.hypothesis_features.shape == (3, 4, 12)
    assert (
        output.metadata["hypothesis_semantics"]
        == "modality_aligned_latent_evidence_slots"
    )
    assert not any(
        token in name
        for name, _ in model.named_modules()
        for token in ("attribution_head", "image_grounding_head", "text_grounding_head")
    )
    output.validate()


def test_binary_hera_real_hgt_path_uses_final_four_family_schema() -> None:
    pytest.importorskip("torch_geometric")
    config = _binary_config()
    assert "face" not in config["evidence"]
    assert "face" not in config["routing"]
    model = BinaryBenchmarkHERA(config, load_backbones=False).eval()

    output = model(_batch(), features=synthetic_features())

    assert output.binary_logit.shape == (3,)
    assert output.metadata["signals"]["image_grounding"] is False
    output.validate()


def test_binary_hera_forward_does_not_read_targets() -> None:
    model = BinaryBenchmarkHERA(
        _binary_config(),
        graph_reasoner=FixtureGraphReasoner(hidden_dim=12),
        load_backbones=False,
    ).eval()
    first_batch = _batch()
    changed = copy.deepcopy(first_batch)
    changed["binary_target"] = 1.0 - changed["binary_target"]
    changed["subtype_target"] = torch.tensor([3, 3, 3])
    changed["text_label_target"] = 1.0 - changed["text_label_target"]

    first = model(first_batch, features=synthetic_features())
    second = model(changed, features=synthetic_features())
    assert torch.allclose(first.binary_logit, second.binary_logit, atol=1e-6)


def test_binary_loss_gates_dataset_native_auxiliary_heads() -> None:
    config = _binary_config(subtype=True, text_label=True)
    model = BinaryBenchmarkHERA(
        config,
        graph_reasoner=FixtureGraphReasoner(hidden_dim=12),
        load_backbones=False,
    )
    losses = BinaryBenchmarkLoss(config)(
        model(_batch(), features=synthetic_features()),
        _batch(),
    )
    assert torch.isfinite(losses.total)
    assert losses.subtype.item() > 0
    assert losses.text_label.item() > 0
    assert SignalContract.from_config(config).subtype is True


def test_binary_threshold_metrics_and_auxiliary_metrics() -> None:
    bundle = BinaryPredictionBundle(
        ids=["a", "b", "c", "d"],
        targets=[0.0, 0.0, 1.0, 1.0],
        scores=[0.1, 0.4, 0.6, 0.9],
        subtype_targets=[0, 1, 2, 3],
        subtype_predictions=[0, 1, 2, 3],
        text_label_targets=[0.0, 0.0, 1.0, 1.0],
        text_label_scores=[0.1, 0.2, 0.8, 0.9],
    )
    threshold = tune_binary_threshold(
        bundle.targets,
        bundle.scores,
        {"binary": {"min": 0.1, "max": 0.9, "step": 0.1}},
    )
    metrics, report = binary_metrics_and_report(
        bundle,
        threshold=threshold,
        negative_name="negative",
        positive_name="positive",
    )
    metrics.update(auxiliary_metrics(bundle))

    assert np.isclose(threshold, 0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["subtype_macro_f1"] == 1.0
    assert metrics["text_label_roc_auc"] == 1.0
    assert "negative" in report and "positive" in report
