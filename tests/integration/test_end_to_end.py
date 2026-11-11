"""Two synthetic examples per ontology through the actual HERI modules."""

import copy
import csv
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.support.baseline import synthetic_batch, synthetic_features
from tests.support.heri import perturb_forbidden_targets
from heri.losses.heri_components import HERAMultitaskLoss
from heri.losses.baseline_losses import BaselineMultitaskLoss
from heri.experiments.ablation import apply_hera_ablation_variant
from heri.evaluation.evaluator import (
    PredictionBundle,
    _accumulate_batch,
    _accumulate_image_ious,
    compute_metrics_and_reports,
)
from heri.evaluation.summary import write_predictions_csv
from heri.thresholding.tuner import Thresholds
from heri.models.heri import HERADGM
from heri.models.baselines.factory import build_baseline_model
from heri.utils.config import load_yaml_config


def _tiny_case(dataset: str):
    config = copy.deepcopy(load_yaml_config(f"config/datasets/{dataset}/heri/train.yaml"))
    for section in ("evidence", "graph", "model"):
        config[section]["hidden_dim"] = 12
        config[section]["dropout"] = 0.0
    config["routing"]["feature_dim"] = 12
    config["routing"]["estimator"]["dropout"] = 0.0
    config["graph"].update(heads=3, layers=2)
    config["model"]["attribution"].update(hidden_dim=12, dropout=0.0)
    config["model"]["grounding"]["image"]["dropout"] = 0.0
    config["model"]["grounding"]["text"].update(token_input_dim=19, dropout=0.0)
    for family in (
        "visual_semantic",
        "visual_forensic",
        "text_alignment",
        "text_context",
    ):
        config["evidence"][family]["max_nodes"] = 2
    batch = synthetic_batch(batch_size=2)
    batch["image"] = torch.rand(2, 3, 32, 32)
    batch["fake_image_box_target"][1] = torch.tensor([2.0, 2.0, 12.0, 12.0])
    batch["text_grounding_target"][1] = torch.tensor([1, 0, 0, -100, -100])
    batch["has_text_manipulation"][1] = True
    batch["fake_text_pos"][1] = torch.tensor([0, -1])
    batch["image_grounding_available"] = torch.ones(2, dtype=torch.bool)
    batch["text_grounding_available"] = torch.ones(2, dtype=torch.bool)
    if dataset == "samm":
        batch["atomic_target"] = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        batch["fake_cls"] = ["orig", "swap_manipulation"]
    else:
        batch["atomic_target"][1] = torch.tensor([1.0, 0.0, 1.0, 0.0])
        batch["fake_cls"][1] = "face_swap&text_swap"
    spans = torch.tensor([[[0, 5], [6, 10], [11, 16], [-1, -1], [-1, -1]]] * 2)
    batch["text_token_spans"] = spans
    features = synthetic_features(batch_size=2)
    features.siglip2.text_word_ids = batch["text_word_ids"]
    features.siglip2.text_token_spans = spans
    features.siglip2.image_grid = replace(
        features.siglip2.image_grid, image_height=32, image_width=32
    )
    features.dinov3.patch_grid = replace(
        features.dinov3.patch_grid, image_height=32, image_width=32
    )
    return config, batch, features


@pytest.mark.parametrize("dataset", ["dgm4", "samm"])
def test_two_samples_real_graph_backward_and_target_independent_inference(dataset, tmp_path):
    torch.manual_seed(7)
    config, batch, features = _tiny_case(dataset)
    model = HERADGM.from_config(config, load_backbones=False)
    model.train()
    output = model(batch, features=features)
    output.validate()
    losses = HERAMultitaskLoss(config["loss"], routing_config=config["routing"])(output, batch)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert output.image_grounding_logits.shape[-1] == 4  # graph keeps only 2
    assert output.text_grounding_logits.shape[-1] == 5
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    saliency = model.evidence_extractor.visual_semantic_extractor.selector.scorer.weight
    assert saliency.grad is not None and saliency.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        clean = model(batch, features=features)
        changed = model(perturb_forbidden_targets(batch), features=features)
    torch.testing.assert_close(clean.atomic_logits, changed.atomic_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        clean.image_grounding_boxes, changed.image_grounding_boxes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        clean.text_grounding_logits, changed.text_grounding_logits, rtol=0, atol=0
    )
    reference_fake = 1 - (1 - clean.atomic_probabilities).prod(dim=1)
    torch.testing.assert_close(clean.binary_probability, reference_fake)
    assert len(clean.metadata["hypothesis_probabilities_by_cycle"]) == 2
    assert clean.graph_metadata["graph_build_count"] == 2

    bundle = PredictionBundle(text_grounding_aggregation="relative_soft")
    _accumulate_batch(bundle, clean, batch)
    names = tuple(config["model"]["hypotheses"]["order"])
    thresholds = Thresholds(
        binary=0.5,
        atomic=(0.5,) * len(names),
        hypothesis_names=names,
        image_presence=0.5,
        image_presence_source="fixed_0.5",
    )
    metrics = compute_metrics_and_reports(bundle, thresholds)["metrics"]
    assert metrics["sample_count"] == 2
    assert metrics["evaluation_protocol"] == "heri_grounded"
    assert metrics["verified_support_metrics_available"]
    assert "vsr_at_1_image" in metrics and "vsr_at_1_text" in metrics
    if dataset == "samm":
        assert metrics["samm_official_decision_threshold"] == 0.5
    path = tmp_path / "predictions.csv"
    write_predictions_csv(
        bundle=bundle,
        thresholds=thresholds,
        path=path,
        run_info={
            "checkpoint_id": "tiny-checkpoint",
            "scientific_study_id": "tiny-study",
            "scientific_variant": "full_hera",
            "seed": 7,
            "evaluation_protocol": "heri_grounded",
            "alignment_objective": "valid_text_mean",
        },
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["hypothesis_probabilities_by_cycle"] for row in rows)
    assert all(row["checkpoint_id"] == "tiny-checkpoint" for row in rows)
    assert all(row["alignment_objective"] == "valid_text_mean" for row in rows)


def test_same_two_examples_cover_component_removals_and_progressive_readouts():
    config, batch, features = _tiny_case("dgm4")
    for variant in (
        "no_forensic_evidence",
        "homogeneous_evidence",
        "no_hypothesis_nodes",
        "generic_grounding",
        "evidence_decomposition",
        "hypothesis_evidence_graph",
    ):
        controlled = apply_hera_ablation_variant(config, variant)
        model = HERADGM.from_config(controlled, load_backbones=False)
        output = model(batch, features=features)
        loss = HERAMultitaskLoss(controlled["loss"], routing_config=controlled["routing"])(
            output, batch
        )
        loss.total.backward()
        assert torch.isfinite(loss.total), variant
        assert all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
        ), variant
        assert output.image_grounding_logits.shape[-1] == 4, variant
        assert model.count_independent_binary_shortcuts() == int(
            controlled["model"]["binary"]["aggregation"] == "independent_head"
        ), variant


def test_backbone_matched_b1_forward_loss_and_unavailable_hypothesis_diagnostics():
    _, batch, features = _tiny_case("samm")
    config = load_yaml_config("config/datasets/samm/baselines/b1/train.yaml")
    config["model"].update(
        fusion_dim=12,
        head_hidden_dim=12,
        num_heads=3,
        num_attention_layers=1,
        dropout=0.0,
    )
    model = build_baseline_model(config, load_backbones=False)
    output = model(batch, features=features)
    loss = BaselineMultitaskLoss(config["loss"]["weights"])(output, batch)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    bundle = PredictionBundle()
    _accumulate_batch(bundle, output, batch)
    metrics = compute_metrics_and_reports(
        bundle,
        Thresholds(
            binary=0.5,
            atomic=(0.5, 0.5),
            hypothesis_names=("SWAP", "ATTRIBUTE"),
            image_presence=0.5,
            image_presence_source="fixed_0.5",
        ),
    )["metrics"]
    assert not metrics["evidence_decision_metrics_available"]
    assert not metrics["verified_support_metrics_available"]


def test_empty_box_abstention_on_manipulated_sample_has_zero_iou():
    output = SimpleNamespace(
        atomic_probabilities=torch.tensor([[0.3, 0.3]]),
        image_grounding_boxes=torch.tensor([[[0.25, 0.25, 0.75, 0.75]] * 2]),
        metadata={
            "image_hypothesis_indices": [0, 1],
            "hypothesis_order": ["SWAP", "ATTRIBUTE"],
        },
    )
    bundle = PredictionBundle()
    batch = {
        "image": torch.zeros(1, 3, 16, 16),
        "fake_image_box_target": torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
        "has_image_manipulation": torch.tensor([True]),
    }
    _accumulate_image_ious(bundle, output, batch)
    assert bundle.image_grounding_pred_ious == [0.0]
    assert bundle.image_ious == [0.0]
