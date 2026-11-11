from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from tests.support.decision import decision_batch
from tests.support.graph import graph_config, synthetic_evidence
from heri.data.collate import multimodal_collate
from heri.data.loader import create_dataset
from heri.data.samm import (
    SAMM_CLASS_ORDER,
    SAMMDataset,
    resolve_samm_image_path,
)
from heri.data.samm_preflight import _ordered_id_manifest_sha256
from heri.evaluation.evaluator import PredictionBundle, compute_metrics_and_reports
from heri.evaluation.summary import write_predictions_csv, write_summary_csv
from heri.experiments.ablation_runner import load_ablation_suite, materialize_ablation_jobs
from heri.graph.builder import HypothesisEvidenceGraphBuilder
from heri.graph.outputs import HypothesisGraphOutput
from heri.graph.validation import graph_statistics
from heri.losses.heri_components import HERAMultitaskLoss, evidence_sufficiency_loss
from heri.models.heri.decision import HERADecisionModule
from heri.thresholding.tuner import Thresholds
from heri.utils.config import (
    ConfigError,
    load_yaml_config,
    validate_baseline_config,
    validate_hera_config,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color=(80, 100, 120)).save(path)


def _tiny_samm(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / "SAMM"
    rows = [
        {
            "id": 1,
            "image": "images/orig_output/1.jpg",
            "text": "original report",
            "fake_cls": "orig",
            "fake_image_box": [],
            "fake_text_pos": [],
        },
        {
            "id": 2,
            "image": "images/swap_jpg/2.jpg",
            "text": "swapped person report",
            "fake_cls": "swap_manipulation",
            "fake_image_box": [-5, 5, 30, 30],
            "fake_text_pos": [0],
        },
        {
            "id": 3,
            "image": "images/emotion_jpg/3.jpg",
            "text": "changed emotion report",
            "fake_cls": "attribute_manipulation",
            "fake_image_box": [10, 5, 90, 45],
            "fake_text_pos": [1],
        },
        {
            "id": 4,
            "image": "images/swap_jpg/4.jpg",
            "text": "missing image annotation",
            "fake_cls": "swap_manipulation",
            "fake_text_pos": [0],
        },
        {
            "id": 5,
            "image": "images/emotion_jpg/5.jpg",
            "text": "bad text index",
            "fake_cls": "attribute_manipulation",
            "fake_image_box": [10, 5, 90, 45],
            "fake_text_pos": [9],
        },
    ]
    for row in rows:
        _write_image(root / row["image"])
    (root / "train.json").write_text(json.dumps(rows), encoding="utf-8")
    return root, rows


def test_samm_dataset_normalizes_coupled_ontology_and_annotation_masks(tmp_path: Path) -> None:
    root, rows = _tiny_samm(tmp_path)
    dataset = SAMMDataset(
        rows=rows,
        split="train",
        dataset_root=root,
        image_size=384,
        verify_schema=True,
        verify_image_exists=True,
        progress=False,
    )
    orig, swap, attribute, missing_box, invalid_text = [dataset[index] for index in range(5)]

    assert orig["atomic_target"].tolist() == [0.0, 0.0]
    assert swap["atomic_target"].tolist() == [1.0, 0.0]
    assert attribute["atomic_target"].tolist() == [0.0, 1.0]
    assert swap["has_image_manipulation"] and swap["has_text_manipulation"]
    assert swap["fake_image_box_target"][0].item() == 0.0
    assert not missing_box["image_grounding_available"]
    assert not invalid_text["text_grounding_available"]
    assert invalid_text["fake_text_pos"].numel() == 0
    assert orig["class_names"] == list(SAMM_CLASS_ORDER)

    batch = multimodal_collate([orig, swap, attribute])
    assert batch["atomic_target"].shape == (3, 2)
    assert batch["class_target"].tolist() == [0, 1, 2]
    assert batch["hypothesis_names"] == ["SWAP", "ATTRIBUTE"]


def test_samm_path_and_loader_dispatch(tmp_path: Path) -> None:
    root, _ = _tiny_samm(tmp_path)
    physical = resolve_samm_image_path(
        "images/swap_jpg/2.jpg", root, fake_cls="swap_manipulation", verify_exists=True
    )
    assert physical == root / "images" / "swap_jpg" / "2.jpg"
    config = {
        "dataset": {
            "name": "SAMM",
            "root": str(root),
            "image_root": str(root / "images"),
            "train_annotations": str(root / "train.json"),
            "verify_image_exists": True,
            "annotation_policy": {"clip_boxes": True, "mask_invalid_text_positions": True},
        },
        "image": {"size": 384},
        "tokenizer": {"max_length": 128, "ignore_index": -100},
        "preflight": {"verify_schema": True},
        "progress": {"enabled": False},
    }
    dataset = create_dataset(config, split="train", samples_per_class=1)
    assert isinstance(dataset, SAMMDataset)
    assert len(dataset) == 3
    assert {row["fake_cls"] for row in dataset.rows} == set(SAMM_CLASS_ORDER)


def test_samm_ordered_id_manifest_is_reproducible_and_order_sensitive() -> None:
    rows = [{"id": 10}, {"id": 20}, {"id": 30}]
    assert _ordered_id_manifest_sha256(rows) == _ordered_id_manifest_sha256(list(rows))
    assert _ordered_id_manifest_sha256(rows) != _ordered_id_manifest_sha256(list(reversed(rows)))


def _samm_decision_fixture() -> tuple[HERADecisionModule, HypothesisGraphOutput, dict]:
    hidden_dim = 12
    evidence = synthetic_evidence(hidden_dim=hidden_dim)
    spec = {
        "order": ["SWAP", "ATTRIBUTE"],
        "image": ["SWAP", "ATTRIBUTE"],
        "text": ["SWAP", "ATTRIBUTE"],
    }
    graph_cfg = graph_config(hidden_dim=hidden_dim)
    graph_cfg["hypotheses"] = spec
    hypotheses = torch.randn(3, 2, hidden_dim, requires_grad=True)
    graph = HypothesisEvidenceGraphBuilder(graph_cfg).build(evidence, hypotheses)
    graph_output = HypothesisGraphOutput(
        hypothesis_features=hypotheses,
        hypothesis_mask=torch.ones(3, 2, dtype=torch.bool),
        graph_batch=graph,
        graph_statistics=graph_statistics(graph),
        updated_node_features=graph.x_dict,
        refined_visual_semantic=evidence.visual_semantic,
        refined_visual_forensic=evidence.visual_forensic,
        refined_text_alignment=evidence.text_alignment,
        refined_text_context=evidence.text_context,
    )
    config = {
        "model": {
            "hidden_dim": hidden_dim,
            "hypotheses": spec,
            "attribution": {"hidden_dim": hidden_dim, "dropout": 0.0},
            "grounding": {
                "mode": "hypothesis_conditioned",
                "routing": "soft",
                "routing_temperature": 1.0,
                "image": {"scoring": "dot_product", "dropout": 0.0},
                "text": {
                    "scoring": "dot_product",
                    "dropout": 0.0,
                    "full_token_store": True,
                    "token_input_dim": 19,
                },
            },
            "binary": {"aggregation": "noisy_or"},
        },
        "loss": {
            "lambda_attr": 1.0,
            "lambda_bin": 0.1,
            "lambda_img": 1.0,
            "lambda_txt": 1.0,
            "lambda_ev": 0.1,
            "lambda_align": 0.0,
            "image_grounding": {"l1_weight": 1.0, "giou_weight": 1.0, "heatmap_weight": 0.25},
        },
    }
    return HERADecisionModule(config), graph_output, config


def test_hera_decision_and_losses_support_two_coupled_samm_hypotheses() -> None:
    module, graph_output, config = _samm_decision_fixture()
    output = module(graph_output)
    assert output.atomic_logits.shape == (3, 2)
    assert output.image_grounding_logits.shape[:2] == (3, 2)
    assert output.text_grounding_logits.shape[:2] == (3, 2)
    assert output.evidence_support.shape == (3, 2)
    assert output.evidence_support_by_modality.shape == (3, 2, 2)
    assert output.evidence_support_modality_mask.tolist() == [
        [True, True],
        [True, True],
    ]
    assert output.metadata["image_hypothesis_indices"] == [0, 1]
    assert output.metadata["text_hypothesis_indices"] == [0, 1]

    batch = decision_batch()
    batch["atomic_target"] = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    batch["has_image_manipulation"] = torch.tensor([False, True, True])
    batch["has_text_manipulation"] = torch.tensor([False, True, True])
    batch["image_grounding_available"] = torch.ones(3, dtype=torch.bool)
    batch["text_grounding_available"] = torch.ones(3, dtype=torch.bool)
    losses = HERAMultitaskLoss(config["loss"])(output, batch)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert torch.isfinite(graph_output.hypothesis_features.grad).all()


def test_samm_per_modality_evidence_loss_exposes_weak_text_support() -> None:
    module, graph_output, _ = _samm_decision_fixture()
    output = module(graph_output)
    support = torch.zeros_like(output.evidence_support_by_modality)
    support[0, 0] = torch.tensor([0.95, 0.05])
    output = replace(
        output,
        atomic_probabilities=torch.tensor([[0.9, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        evidence_support_by_modality=support,
        evidence_support=support.min(dim=-1).values,
    )
    batch = decision_batch()
    batch["atomic_target"] = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    batch["image_grounding_available"] = torch.ones(3, dtype=torch.bool)
    batch["text_grounding_available"] = torch.ones(3, dtype=torch.bool)

    per_modality = evidence_sufficiency_loss(output, batch)
    legacy_or = evidence_sufficiency_loss(output, batch, aggregation="any_modality_or")

    assert torch.isclose(per_modality, torch.tensor(0.4250))
    assert torch.isclose(legacy_or, torch.tensor(0.0))


def test_samm_per_modality_evidence_loss_masks_unavailable_annotations() -> None:
    module, graph_output, _ = _samm_decision_fixture()
    output = module(graph_output)
    support = torch.zeros_like(output.evidence_support_by_modality)
    support[0, 0] = torch.tensor([0.95, 0.05])
    output = replace(
        output,
        atomic_probabilities=torch.tensor([[0.9, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        evidence_support_by_modality=support,
        evidence_support=support.min(dim=-1).values,
    )
    batch = decision_batch()
    batch["atomic_target"] = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    batch["image_grounding_available"] = torch.ones(3, dtype=torch.bool)
    batch["text_grounding_available"] = torch.tensor([False, True, True])

    loss = evidence_sufficiency_loss(output, batch)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_samm_per_modality_evidence_loss_empty_domain_is_differentiable_zero() -> None:
    module, graph_output, _ = _samm_decision_fixture()
    output = module(graph_output)
    batch = decision_batch()
    batch["atomic_target"] = torch.zeros_like(batch["atomic_target"][:, :2])

    loss = evidence_sufficiency_loss(output, batch)

    assert loss.item() == 0.0
    assert loss.requires_grad


def test_samm_metrics_include_three_class_and_released_fixed_threshold_result(
    tmp_path: Path,
) -> None:
    bundle = PredictionBundle(
        ids=[1, 2, 3],
        fake_cls=list(SAMM_CLASS_ORDER),
        has_image_manipulation=[False, True, True],
        has_text_manipulation=[False, True, True],
        binary_targets=[0.0, 1.0, 1.0],
        binary_scores=[0.1, 0.9, 0.8],
        atomic_targets=[[0, 0], [1, 0], [0, 1]],
        atomic_scores=[[0.1, 0.1], [0.9, 0.1], [0.2, 0.8]],
        class_targets=[0, 1, 2],
        class_names=list(SAMM_CLASS_ORDER),
        hypothesis_names=["SWAP", "ATTRIBUTE"],
    )
    thresholds = Thresholds(binary=0.5, atomic=(0.5, 0.5), hypothesis_names=("SWAP", "ATTRIBUTE"))
    reports = compute_metrics_and_reports(bundle, thresholds)
    assert reports["metrics"]["class_accuracy"] == 1.0
    assert reports["metrics"]["class_macro_f1"] == 1.0
    assert reports["metrics"]["samm_official_map"] == 1.0
    assert reports["metrics"]["samm_official_cf1"] == 1.0
    assert reports["metrics"]["samm_official_of1"] == 1.0
    assert reports["metrics"]["samm_official_op"] == 1.0
    assert reports["metrics"]["samm_official_or"] == 1.0
    assert reports["metrics"]["samm_official_cp"] == 1.0
    assert reports["metrics"]["samm_official_cr"] == 1.0
    assert reports["metrics"]["samm_official_exact_match_ratio"] == 1.0
    assert reports["metrics"]["samm_official_target_mapping"] == (
        "orig=000;swap_manipulation=101;attribute_manipulation=011"
    )
    assert reports["metrics"]["samm_official_reference_commit"] == (
        "f685f0d6e64dbbbc5a4e787eab768f5f42099ce9"
    )
    assert not reports["metrics"]["evidence_decision_metrics_available"]
    assert reports["metrics"]["evidence_decision_consistency"] is None
    assert reports["metrics"]["unsupported_decision_rate"] is None
    assert reports["metrics"]["edc"] is None
    assert reports["metrics"]["udr"] is None
    assert reports["samm_official_report_dict"] is not None
    assert set(reports["attribution_report_dict"]) >= {"swap", "attribute"}
    assert thresholds.as_dict()["swap_threshold"] == 0.5
    predictions = tmp_path / "predictions.csv"
    summary = tmp_path / "summary.csv"
    write_predictions_csv(bundle=bundle, thresholds=thresholds, path=predictions)
    write_summary_csv(
        path=summary,
        run_info={"dataset_name": "SAMM"},
        metrics=reports["metrics"],
        thresholds=thresholds,
        reports=reports,
    )
    prediction_frame = pd.read_csv(predictions)
    summary_frame = pd.read_csv(summary)
    assert prediction_frame["predicted_class_name"].tolist() == list(SAMM_CLASS_ORDER)
    assert "orig_decision_score" in prediction_frame
    assert "samm_official_text_attribute_probability" in prediction_frame
    assert "class_attribute_manipulation_f1" in summary_frame
    assert "samm_official_visual_swap_f1" in summary_frame


def test_samm_released_image_metric_retains_missing_box_rows() -> None:
    bundle = PredictionBundle(
        ids=[1, 2, 3],
        fake_cls=list(SAMM_CLASS_ORDER),
        has_image_manipulation=[False, True, True],
        has_text_manipulation=[False, True, True],
        image_grounding_available=[True, True, False],
        text_grounding_available=[True, True, True],
        binary_targets=[0.0, 1.0, 1.0],
        binary_scores=[0.1, 0.9, 0.8],
        atomic_targets=[[0, 0], [1, 0], [0, 1]],
        atomic_scores=[[0.1, 0.1], [0.9, 0.1], [0.2, 0.8]],
        hypothesis_names=["SWAP", "ATTRIBUTE"],
        image_presence_scores=[0.1, 0.9, 0.9],
        image_grounding_pred_boxes=[
            "[0.2, 0.2, 0.4, 0.4]",
            "[0.2, 0.2, 0.4, 0.4]",
            "[0.2, 0.2, 0.4, 0.4]",
        ],
        image_grounding_pred_ious=[0.0, 0.8, None],
    )
    reports = compute_metrics_and_reports(
        bundle,
        Thresholds(
            binary=0.5,
            atomic=(0.5, 0.5),
            hypothesis_names=("SWAP", "ATTRIBUTE"),
            image_presence=0.5,
            image_presence_source="fixed_0.5",
        ),
    )
    assert reports["metrics"]["samm_released_all_sample_image_count"] == 3
    assert reports["metrics"]["samm_released_all_sample_image_mean_iou"] == pytest.approx(0.6)


def test_all_samm_configs_resolve_and_validate() -> None:
    for path, mode in (
        ("config/datasets/samm/heri/train.yaml", "train"),
        ("config/datasets/samm/heri/evaluate.yaml", "evaluate"),
    ):
        config = load_yaml_config(path)
        assert config["tokenizer"]["max_length"] == 128
        assert config["backbones"]["siglip2"]["text_max_length"] == 64
        assert config["model"]["grounding"]["text"]["aggregation"] == "relative_soft"
        validate_hera_config(config, mode=mode)
    for path, mode in (
        ("config/datasets/samm/baselines/b1/train.yaml", "train"),
        ("config/datasets/samm/baselines/b1/evaluate.yaml", "evaluate"),
    ):
        config = load_yaml_config(path)
        assert config["tokenizer"]["max_length"] == 128
        assert config["backbones"]["siglip2"]["text_max_length"] == 64
        validate_baseline_config(config, mode=mode)


def test_samm_scientific_config_rejects_tuned_thresholds_and_early_stop() -> None:
    config = load_yaml_config("config/datasets/samm/heri/train.yaml")
    config["thresholding"]["enabled"] = True
    with pytest.raises(ConfigError, match="fixed 0.5"):
        validate_hera_config(config, mode="train")

    config = load_yaml_config("config/datasets/samm/heri/train.yaml")
    config["training"]["early_stopping"]["enabled"] = True
    with pytest.raises(ConfigError, match="early stopping"):
        validate_hera_config(config, mode="train")


def test_samm_key_ablation_suite_materializes_only_prespecified_variants(
    tmp_path: Path,
) -> None:
    suite = load_ablation_suite("config/experiments/samm/key_ablations.yaml")
    suite["output_root"] = str(tmp_path / "samm_ablations")
    jobs = materialize_ablation_jobs(suite, repo_root=Path.cwd())
    assert [job.variant for job in jobs] == ["no_router", "generic_grounding"]
    for job in jobs:
        train_config = load_yaml_config(job.train_config_path)
        evaluate_config = load_yaml_config(job.evaluate_config_path)
        assert train_config["dataset"]["name"] == "SAMM"
        assert train_config["model"]["hypotheses"]["order"] == ["SWAP", "ATTRIBUTE"]
        assert train_config["project"]["name"].startswith("HERA-SAMM-ablation-")
        validate_hera_config(train_config, mode="train")
        validate_hera_config(evaluate_config, mode="evaluate")
