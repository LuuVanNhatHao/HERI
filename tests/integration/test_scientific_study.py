import copy
import json
from pathlib import Path
from tests.support.paths import REPO_ROOT
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.support.graph import graph_config, synthetic_evidence
from tests.support.baseline import synthetic_batch, synthetic_features
from heri.evaluation.evaluator import (
    PredictionBundle,
    _accumulate_text_counts,
    _accumulate_image_ious,
    _gated_image_ious,
    _image_metrics_from_ious,
)
from heri.evaluation.protocol import EVALUATION_PROTOCOL
from heri.evaluation.hypothesis_grounding import image_grounding_presence_probability
from heri.evaluation.run import _load_thresholds
from heri.experiments.scientific_study import aggregate_study, materialize_study, mean_interval
from heri.graph.query_reasoner import LearnedQueryReasoner
from heri.losses.baseline_losses import BaselineMultitaskLoss
from heri.metrics.attribution import attribution_metrics
from heri.models.baselines.b1_cross_attention import B1CrossAttentionBaseline
from heri.training.checkpoint import restore_checkpoint
from heri.evaluation.interventions import rectangle_masks, deleted_batch
from tests.support.heri import hera_test_config, perturb_forbidden_targets
from heri.models.heri import HERADGM


def test_cf1_uses_harmonic_class_means_not_macro_f1():
    true = np.array([[1, 1], [0, 1], [0, 1]])
    score = np.array([[0.9, 0.9], [0.9, 0.1], [0.9, 0.1]])
    result = attribution_metrics(true, score, thresholds=np.array([0.5, 0.5]))
    assert result["cf1"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(0.5)


def test_all_sample_text_counts_include_real_false_positives_and_truncated_targets():
    output = SimpleNamespace(
        text_grounding_logits=torch.tensor([[2.0, -2.0], [2.0, -2.0]]),
        text_grounding_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    batch = {
        "text_grounding_target": torch.tensor([[0, 0], [1, 0]]),
        "text_word_ids": torch.tensor([[0, 1], [0, 1]]),
        "has_text_manipulation": torch.tensor([False, True]),
        "fake_text_pos": torch.tensor([[-1, -1], [0, 3]]),
        "words": [["real", "caption"], ["a", "b", "c", "unseen"]],
    }
    bundle = PredictionBundle()
    _accumulate_text_counts(bundle, output, batch)
    assert (bundle.text_tp, bundle.text_fp, bundle.text_fn) == (1, 1, 1)
    assert (bundle.text_token_tp, bundle.text_token_fp, bundle.text_token_fn) == (1, 1, 0)
    assert bundle.text_unseen_positive_words == 1
    assert bundle.text_total_words == 6
    assert bundle.text_positive_fp == 0


def test_image_all_scope_and_strict_iou_threshold():
    output = SimpleNamespace(
        image_grounding_boxes=torch.tensor(
            [[0.1, 0.1, 0.8, 0.8], [0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.8, 0.8]]
        ),
        atomic_probabilities=torch.ones(3, 4) * 0.5,
    )
    batch = {
        "image": torch.zeros(3, 3, 10, 10),
        "has_image_manipulation": torch.tensor([False, False, True]),
        "fake_image_box_target": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 8.0, 8.0]]
        ),
    }
    bundle = PredictionBundle()
    _accumulate_image_ious(bundle, output, batch)
    assert bundle.image_all_ious == pytest.approx([0.0, 1.0, 1.0])
    assert bundle.image_ious == pytest.approx([1.0])
    assert _image_metrics_from_ious([0.5, 0.75])["image_iou50"] == 0.5
    assert _image_metrics_from_ious([0.5, 0.75])["image_iou75"] == 0


def test_image_presence_is_ontology_driven_and_abstains_without_targets():
    dgm4 = SimpleNamespace(
        atomic_probabilities=torch.tensor([[0.2, 0.5, 0.9, 0.8]]),
        metadata={"image_hypothesis_indices": [0, 1]},
    )
    samm = SimpleNamespace(
        atomic_probabilities=torch.tensor([[0.2, 0.5]]),
        metadata={"image_hypothesis_indices": [0, 1]},
    )
    assert image_grounding_presence_probability(dgm4).item() == pytest.approx(0.6)
    assert image_grounding_presence_probability(samm).item() == pytest.approx(0.6)

    bundle = PredictionBundle(
        ids=[1, 2, 3],
        has_image_manipulation=[False, True, True],
        image_grounding_available=[True, True, True],
        image_presence_scores=[0.1, 0.2, 0.9],
        image_grounding_pred_ious=[0.0, 0.8, 0.7],
        image_all_ious=[0.0, 0.8, 0.7],
        image_ious=[0.8, 0.7],
    )
    all_ious, positive_ious = _gated_image_ious(bundle, 0.5)
    assert all_ious == pytest.approx([1.0, 0.0, 0.7])
    assert positive_ious == pytest.approx([0.0, 0.7])
    assert _image_metrics_from_ious(all_ious)["image_iou95"] == pytest.approx(1 / 3)


def test_query_control_has_no_edges_and_backward_is_finite():
    evidence = synthetic_evidence()
    model = LearnedQueryReasoner({"graph": graph_config(cycles=2)}).eval()
    output = model(evidence)
    assert output.hypothesis_features.shape == (3, 4, 12)
    assert all(edge.numel() == 0 for edge in output.graph_batch.edge_index_dict.values())
    output.hypothesis_features.square().sum().backward()
    assert torch.isfinite(model.hypotheses.embeddings.grad).all()
    before = output.hypothesis_features.detach().clone()
    changed = copy.copy(evidence)
    changed.visual_semantic = copy.copy(evidence.visual_semantic)
    changed.visual_semantic.features = evidence.visual_semantic.features.detach().clone()
    changed.visual_semantic.features[0] += 5
    after = model(changed).hypothesis_features.detach()
    assert torch.allclose(before[1:], after[1:])
    assert not torch.allclose(before[0], after[0])


def test_b1_noisy_or_has_no_shortcut_and_masks_absent_fake_gradients():
    model = B1CrossAttentionBaseline(
        fusion_dim=12,
        num_layers=1,
        num_heads=3,
        head_hidden_dim=12,
        dropout=0.0,
        binary_aggregation="noisy_or",
    )
    batch = synthetic_batch()
    output = model(batch, features=synthetic_features())
    assert not hasattr(model, "binary_head")
    assert torch.allclose(
        torch.sigmoid(output.binary_logit),
        1 - (1 - torch.sigmoid(output.atomic_logits)).prod(-1),
        atol=1e-6,
    )
    output.atomic_logits.retain_grad()
    result = BaselineMultitaskLoss(
        {"binary": 1, "attribution": 0, "image_grounding": 0, "text_grounding": 0}
    )(output, batch)
    result.binary.backward()
    absent_fake = (batch["binary_target"] == 1).unsqueeze(-1) & (batch["atomic_target"] == 0)
    assert torch.equal(
        output.atomic_logits.grad[absent_fake],
        torch.zeros_like(output.atomic_logits.grad[absent_fake]),
    )


def test_query_control_full_forward_is_target_independent_and_keeps_dense_stores():
    config = hera_test_config()
    config["reasoning_control"] = {"enabled": True, "variant": "learned_queries"}
    config["graph"]["reasoning"].update(cycles=2, dynamic_rebuild=False)
    model = HERADGM(config, load_backbones=False).eval()
    batch, features = synthetic_batch(), synthetic_features()
    first = model(batch, features=features)
    second = model(perturb_forbidden_targets(batch), features=features)
    assert torch.allclose(first.atomic_logits, second.atomic_logits)
    assert torch.allclose(first.image_grounding_boxes, second.image_grounding_boxes)
    assert torch.allclose(first.text_grounding_logits, second.text_grounding_logits)
    assert first.text_grounding_logits.shape[-1] == features.deberta.token_features.shape[1]
    assert "LearnedQueries" in first.metadata["pipeline"]


def test_legacy_resume_rejected_before_weight_mutation():
    model = torch.nn.Linear(1, 1)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="legacy"):
        restore_checkpoint(
            checkpoint={
                "model": {
                    "weight": torch.zeros_like(model.weight),
                    "bias": torch.zeros_like(model.bias),
                }
            },
            model=model,
        )
    assert torch.equal(before, model.weight)


def test_checkpoint_identity_rejects_same_path_different_weights(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"checkpoint_id": "old", "checkpoint_path": "best.pt"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity"):
        _load_thresholds(path, checkpoint_path=Path("best.pt"), checkpoint={"checkpoint_id": "new"})


def test_deletion_controls_preserve_area_padding_and_targets():
    content = torch.zeros(2, 10, 10, dtype=torch.bool)
    content[:, 1:9, 2:8] = True
    boxes = torch.tensor([[0.1, 0.1, 0.5, 0.7], [0.0, 0.0, 0.0, 0.0]])
    generator = torch.Generator().manual_seed(1)
    selected = rectangle_masks(boxes, content, generator=generator)
    random_mask = rectangle_masks(boxes, content, generator=generator, randomize=True)
    assert torch.equal(selected.sum((-1, -2)), random_mask.sum((-1, -2)))
    assert not bool((random_mask & ~content).any())
    batch = {
        "image": torch.rand(2, 3, 10, 10),
        "image_content_mask": content,
        "atomic_target": torch.ones(2, 4),
    }
    changed = deleted_batch(batch, selected)
    assert torch.equal(
        changed["image"].permute(0, 2, 3, 1)[~selected],
        batch["image"].permute(0, 2, 3, 1)[~selected],
    )
    assert changed["atomic_target"] is batch["atomic_target"]


def test_matched_study_pending_tables_and_seed_intervals(tmp_path):
    root = REPO_ROOT
    suite = {
        "seeds": [7, 21, 101],
        "variants": ["b1", "b1_noisy_or", "learned_queries", "two_cycle_static", "full_hera"],
        "epochs": 20,
        "hera_train_config": "config/datasets/dgm4/heri/train.yaml",
        "b1_train_config": "config/datasets/dgm4/baselines/b1/train.yaml",
        "output_root": str(tmp_path),
    }
    manifest = materialize_study(suite, root)
    assert len(manifest["jobs"]) == 15
    assert aggregate_study(manifest)["status"] == "pending"
    interval = mean_interval([1, 2, 3])
    assert interval["mean"] == 2
    assert interval["ci95_low"] == pytest.approx(2 - 4.3026527299 / np.sqrt(3))
    assert mean_interval([1])["ci95_low"] is None


def test_aggregator_pairs_by_seed_and_rejects_legacy_or_partial_results(tmp_path):
    import csv
    from heri.experiments.scientific_study import METRICS

    manifest = {
        "protocol_id": "fixture",
        "seeds": [7, 21, 101],
        "expected_test_samples": 100,
        "jobs": [],
    }
    for variant in ("b1", "full_hera"):
        for seed in manifest["seeds"]:
            path = tmp_path / f"{variant}_{seed}.csv"
            row = {
                "scientific_study_id": "fixture",
                "seed": seed,
                "scientific_variant": variant,
                "split": "test",
                "checkpoint_id": "fixture-weight",
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "checkpoint_evaluation_protocol": EVALUATION_PROTOCOL,
                "sample_count": 100,
                "text_coverage_complete": True,
                **{
                    metric: 0.5 + seed / 10000 + (0.1 if variant == "full_hera" else 0)
                    for metric in METRICS
                },
            }
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            manifest["jobs"].append({"variant": variant, "seed": seed, "summary": str(path)})
    report = aggregate_study(manifest)
    assert report["status"] == "complete"
    assert report["paired_full_minus_control"]["b1"]["cf1_fixed"]["mean"] == pytest.approx(0.1)
    path.write_text(path.read_text().replace(EVALUATION_PROTOCOL, "legacy"), encoding="utf-8")
    with pytest.raises(ValueError, match="Unmatched"):
        aggregate_study(manifest)
