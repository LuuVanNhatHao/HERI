from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from decision_fixtures import decision_batch
from graph_fixtures import graph_config, synthetic_evidence
from src.data.collate import multimodal_collate
from src.data.loader import create_dataset
from src.data.samm import (
    SAMM_CLASS_ORDER,
    SAMMDataset,
    resolve_samm_image_path,
)
from src.evaluation.evaluator import PredictionBundle, compute_metrics_and_reports
from src.evaluation.summary import write_predictions_csv, write_summary_csv
from src.experiments.ablation_runner import load_ablation_suite, materialize_ablation_jobs
from src.graph.builder import HypothesisEvidenceGraphBuilder
from src.graph.outputs import HypothesisGraphOutput
from src.graph.validation import graph_statistics
from src.losses.hera_components import HERAMultitaskLoss
from src.models.hera.decision import HERADecisionModule
from src.thresholding.tuner import Thresholds
from src.utils.config import load_yaml_config, validate_baseline_config, validate_hera_config


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), color=(80, 100, 120)).save(path)


def _tiny_samm(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / "SAMM"
    rows = [
        {"id": 1, "image": "images/orig_output/1.jpg", "text": "original report",
         "fake_cls": "orig", "fake_image_box": [], "fake_text_pos": []},
        {"id": 2, "image": "images/swap_jpg/2.jpg", "text": "swapped person report",
         "fake_cls": "swap_manipulation", "fake_image_box": [-5, 5, 30, 30], "fake_text_pos": [0]},
        {"id": 3, "image": "images/emotion_jpg/3.jpg", "text": "changed emotion report",
         "fake_cls": "attribute_manipulation", "fake_image_box": [10, 5, 90, 45], "fake_text_pos": [1]},
        {"id": 4, "image": "images/swap_jpg/4.jpg", "text": "missing image annotation",
         "fake_cls": "swap_manipulation", "fake_text_pos": [0]},
        {"id": 5, "image": "images/emotion_jpg/5.jpg", "text": "bad text index",
         "fake_cls": "attribute_manipulation", "fake_image_box": [10, 5, 90, 45], "fake_text_pos": [9]},
    ]
    for row in rows:
        _write_image(root / row["image"])
    (root / "train.json").write_text(json.dumps(rows), encoding="utf-8")
    return root, rows


def test_samm_dataset_normalizes_coupled_ontology_and_annotation_masks(tmp_path: Path) -> None:
    root, rows = _tiny_samm(tmp_path)
    dataset = SAMMDataset(
        rows=rows, split="train", dataset_root=root, image_size=384,
        verify_schema=True, verify_image_exists=True, progress=False,
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
            "name": "SAMM", "root": str(root), "image_root": str(root / "images"),
            "train_annotations": str(root / "train.json"), "verify_image_exists": True,
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


def _samm_decision_fixture() -> tuple[HERADecisionModule, HypothesisGraphOutput, dict]:
    hidden_dim = 12
    evidence = synthetic_evidence(hidden_dim=hidden_dim)
    spec = {"order": ["SWAP", "ATTRIBUTE"], "image": ["SWAP", "ATTRIBUTE"],
            "text": ["SWAP", "ATTRIBUTE"]}
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
                "mode": "hypothesis_conditioned", "routing": "soft", "routing_temperature": 1.0,
                "image": {"scoring": "dot_product", "dropout": 0.0},
                "text": {"scoring": "dot_product", "dropout": 0.0,
                         "full_token_store": True, "token_input_dim": 19},
            },
            "binary": {"aggregation": "noisy_or"},
        },
        "loss": {"lambda_attr": 1.0, "lambda_bin": 0.1, "lambda_img": 1.0,
                 "lambda_txt": 1.0, "lambda_ev": 0.1, "lambda_align": 0.0,
                 "image_grounding": {"l1_weight": 1.0, "giou_weight": 1.0, "heatmap_weight": 0.25}},
    }
    return HERADecisionModule(config), graph_output, config


def test_hera_decision_and_losses_support_two_coupled_samm_hypotheses() -> None:
    module, graph_output, config = _samm_decision_fixture()
    output = module(graph_output)
    assert output.atomic_logits.shape == (3, 2)
    assert output.image_grounding_logits.shape[:2] == (3, 2)
    assert output.text_grounding_logits.shape[:2] == (3, 2)
    assert output.evidence_support.shape == (3, 2)
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


def test_samm_metrics_include_three_class_result_and_dynamic_thresholds(
    tmp_path: Path,
) -> None:
    bundle = PredictionBundle(
        ids=[1, 2, 3], fake_cls=list(SAMM_CLASS_ORDER),
        has_image_manipulation=[False, True, True],
        has_text_manipulation=[False, True, True],
        binary_targets=[0.0, 1.0, 1.0], binary_scores=[0.1, 0.9, 0.8],
        atomic_targets=[[0, 0], [1, 0], [0, 1]],
        atomic_scores=[[0.1, 0.1], [0.9, 0.1], [0.2, 0.8]],
        class_targets=[0, 1, 2], class_names=list(SAMM_CLASS_ORDER),
        hypothesis_names=["SWAP", "ATTRIBUTE"],
    )
    thresholds = Thresholds(
        binary=0.5, atomic=(0.5, 0.5), hypothesis_names=("SWAP", "ATTRIBUTE")
    )
    reports = compute_metrics_and_reports(bundle, thresholds)
    assert reports["metrics"]["class_accuracy"] == 1.0
    assert reports["metrics"]["class_macro_f1"] == 1.0
    assert set(reports["attribution_report_dict"]) >= {"swap", "attribute"}
    assert thresholds.as_dict()["swap_threshold"] == 0.5
    predictions = tmp_path / "predictions.csv"
    summary = tmp_path / "summary.csv"
    write_predictions_csv(bundle=bundle, thresholds=thresholds, path=predictions)
    write_summary_csv(
        path=summary, run_info={"dataset_name": "SAMM"},
        metrics=reports["metrics"], thresholds=thresholds, reports=reports,
    )
    prediction_frame = pd.read_csv(predictions)
    summary_frame = pd.read_csv(summary)
    assert prediction_frame["predicted_class_name"].tolist() == list(SAMM_CLASS_ORDER)
    assert "orig_decision_score" in prediction_frame
    assert "class_attribute_manipulation_f1" in summary_frame


def test_all_samm_configs_resolve_and_validate() -> None:
    for path, mode in (
        ("config/train_samm_hera.yaml", "train"),
        ("config/evaluate_samm_hera.yaml", "evaluate"),
        ("config/evaluate_samm_hera_text_grounding_diagnostic.yaml", "evaluate"),
        ("config/smoke_samm_hera.yaml", "smoke"),
    ):
        config = load_yaml_config(path)
        assert config["tokenizer"]["max_length"] == 128
        assert config["backbones"]["siglip2"]["text_max_length"] == 64
        validate_hera_config(config, mode=mode)
    for path, mode in (
        ("config/train_samm_baseline_b1.yaml", "train"),
        ("config/evaluate_samm_baseline_b1.yaml", "evaluate"),
        ("config/smoke_samm_baseline_b1.yaml", "smoke"),
    ):
        config = load_yaml_config(path)
        assert config["tokenizer"]["max_length"] == 128
        assert config["backbones"]["siglip2"]["text_max_length"] == 64
        validate_baseline_config(config, mode=mode)


def test_samm_key_ablation_suite_materializes_only_prespecified_variants(
    tmp_path: Path,
) -> None:
    suite = load_ablation_suite("config/experiments/hera_samm_key_ablations.yaml")
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
