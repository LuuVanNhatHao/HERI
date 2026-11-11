"""Small artifact chains exercise provenance checks without loading checkpoints."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from heri.experiments.provenance import (
    canonical_config_sha256,
    checkpoint_metadata_path,
    inspect_run_evidence,
    inspect_study_evidence,
    write_evaluation_evidence,
)


SOURCE = "a" * 64


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def artifact_run(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "run" / "train" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque tiny checkpoint; never unpickled")
    evaluation = tmp_path / "run" / "evaluation"
    annotations = tmp_path / "data" / "test.json"
    _json(annotations, [{"id": 1}, {"id": 2}])
    annotation_hash = hashlib.sha256(annotations.read_bytes()).hexdigest()
    study = {"protocol_id": "tiny-study", "variant": "evidence_decomposition", "seed": 7}
    train = {
        "project": {"seed": 7, "run_id": "train", "architecture_version": "heri"},
        "model": {"model_type": "hera", "hidden_dim": 2},
        "dataset": {"test_annotations": str(annotations)},
        "dataset_hashes": {"test": annotation_hash},
        "loss": {"alignment": {"aggregation": "valid_text_mean"}},
        "training": {"resume_from": None, "epochs": 20},
        "scientific_study": study,
        "evaluation": {"checkpoint_path": str(checkpoint), "split": "test"},
    }
    evaluate = copy.deepcopy(train)
    evaluate["project"]["run_id"] = "evaluation"
    train_path, evaluate_path = tmp_path / "configs/train.yaml", tmp_path / "configs/evaluate.yaml"
    _yaml(train_path, train)
    _yaml(evaluate_path, evaluate)
    train_effective = copy.deepcopy(train)
    train_effective["observed_dataset_hashes"] = {"train": "train-observed"}
    evaluate_effective = copy.deepcopy(evaluate)
    evaluate_effective["observed_dataset_hashes"] = {"test": annotation_hash}
    _yaml(checkpoint.parent.parent / "effective_config.yaml", train_effective)
    _yaml(evaluation / "effective_config.yaml", evaluate_effective)
    environment = {"source_sha256": SOURCE, "git_commit": "synthetic-source"}
    _json(checkpoint.parent.parent / "environment.json", environment)
    _json(evaluation / "environment.json", environment)
    _json(
        checkpoint_metadata_path(checkpoint),
        {
            "schema": "heri_checkpoint_metadata",
            "checkpoint_id": "checkpoint-1",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "effective_config_sha256": canonical_config_sha256(train_effective),
            "effective_config": train_effective,
            "evaluation_protocol": "heri_grounded",
            "alignment_objective": "valid_text_mean",
            "source_sha256": SOURCE,
        },
    )
    identity = {
        "checkpoint_id": "checkpoint-1",
        "scientific_study_id": study["protocol_id"],
        "scientific_variant": study["variant"],
        "seed": 7,
        "evaluation_protocol": "heri_grounded",
        "alignment_objective": "valid_text_mean",
    }
    summary = evaluation / "summary.csv"
    predictions = evaluation / "predictions/test_predictions.csv"
    _csv(
        summary,
        [
            {
                **identity,
                "checkpoint_evaluation_protocol": "heri_grounded",
                "split": "test",
                "sample_count": 2,
                "dataset_hashes": json.dumps({"test": annotation_hash}),
            }
        ],
    )
    _csv(
        predictions,
        [
            {"id": index, "binary_probability": value, **identity}
            for index, value in ((1, 0.25), (2, 0.75))
        ],
    )
    manifest = tmp_path / "manifest.json"
    _json(
        manifest,
        {
            "protocol_id": study["protocol_id"],
            "evaluation_protocol": "heri_grounded",
            "source_digest": SOURCE,
            "expected_test_samples": 2,
            "jobs": [
                {
                    "variant": study["variant"],
                    "seed": 7,
                    "train_config": str(train_path),
                    "evaluate_config": str(evaluate_path),
                    "train_config_hash": canonical_config_sha256(train),
                    "evaluate_config_hash": canonical_config_sha256(evaluate),
                    "summary": str(summary),
                }
            ],
        },
    )
    write_evaluation_evidence(
        run_dir=evaluation, checkpoint_path=checkpoint, checkpoint_id="checkpoint-1"
    )
    return {
        "manifest": manifest,
        "checkpoint": checkpoint,
        "evaluation": evaluation,
        "summary": summary,
        "predictions": predictions,
        "annotations": annotations,
    }


def _inspect(run: dict) -> dict:
    return inspect_study_evidence(run["manifest"], repository_root=run["manifest"].parent)


def _reseal(run: dict) -> None:
    write_evaluation_evidence(
        run_dir=run["evaluation"], checkpoint_path=run["checkpoint"], checkpoint_id="checkpoint-1"
    )


def test_real_checkpoint_writer_and_audit_share_the_same_artifact_contract(artifact_run):
    import torch
    from heri.training.checkpoint import save_checkpoint
    from heri.training.state import TrainingState
    from heri.utils.config import load_yaml_config

    run = artifact_run
    model = torch.nn.Linear(1, 1)
    checkpoint_id = save_checkpoint(
        path=run["checkpoint"],
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        scaler=None,
        state=TrainingState(),
        metrics={},
        config=load_yaml_config(run["checkpoint"].parent.parent / "effective_config.yaml"),
        precision="fp32",
    )
    for path in (run["summary"], run["predictions"]):
        rows = _csv_rows(path)
        for row in rows:
            row["checkpoint_id"] = checkpoint_id
        _csv(path, rows)
    write_evaluation_evidence(
        run_dir=run["evaluation"], checkpoint_path=run["checkpoint"], checkpoint_id=checkpoint_id
    )
    assert _inspect(run)["status"] == "internally_consistent"


def test_scientific_aggregation_requires_artifacts_not_only_matching_summary_labels(artifact_run):
    from heri.experiments.scientific_study import METRICS, aggregate_study

    run = artifact_run
    rows = _csv_rows(run["summary"])
    rows[0].update({metric: 0.5 for metric in METRICS})
    rows[0]["text_original_word_metadata_complete"] = True
    _csv(run["summary"], rows)
    _reseal(run)
    manifest = json.loads(run["manifest"].read_text())
    manifest.update(evidence_policy="artifact_consistency", seeds=[7])
    report = aggregate_study(manifest)
    assert report["status"] == "complete"
    run["checkpoint"].unlink()
    report = aggregate_study(manifest)
    assert report["status"] == "pending"
    assert report["variants"]["evidence_decomposition"] == {"status": "pending"}


def test_complete_artifact_chain_is_internally_consistent_without_checkpoint_loading(
    artifact_run: dict,
) -> None:
    report = _inspect(artifact_run)
    assert report["status"] == "internally_consistent", report
    assert report["counts"] == {
        "total": 1,
        "internally_consistent": 1,
        "missing": 0,
        "inconsistent": 0,
    }
    assert report["scope"] == "file_integrity_and_run_identity"
    assert report["jobs"][0]["artifacts"]["checkpoint"]["sha256"]
    assert "metrics" not in report["jobs"][0]


def test_missing_checkpoint_is_missing_evidence_not_a_completed_run(artifact_run: dict) -> None:
    artifact_run["checkpoint"].unlink()
    report = _inspect(artifact_run)
    assert report["status"] == "missing", report
    assert any("checkpoint:" in item for item in report["jobs"][0]["missing"])


def test_a_plan_and_named_summary_do_not_establish_run_evidence(artifact_run: dict) -> None:
    (artifact_run["evaluation"] / "run_evidence.json").unlink()
    report = _inspect(artifact_run)
    assert report["status"] == "missing"
    assert any("evaluation_evidence" in item for item in report["jobs"][0]["missing"])


@pytest.mark.parametrize(
    "mutation",
    (
        "checkpoint",
        "config",
        "prediction_count",
        "prediction_id",
        "prediction_identity",
        "summary_identity",
        "annotations",
        "source",
    ),
)
def test_inconsistent_files_fail_even_with_a_fresh_completion_seal(
    artifact_run: dict, mutation: str
) -> None:
    run = artifact_run
    if mutation == "checkpoint":
        run["checkpoint"].write_bytes(b"different checkpoint bytes")
    elif mutation == "config":
        path = run["checkpoint"].parent.parent / "effective_config.yaml"
        config = yaml.safe_load(path.read_text())
        config["model"]["hidden_dim"] = 4
        _yaml(path, config)
    elif mutation in {"prediction_count", "prediction_id", "prediction_identity"}:
        rows = _csv_rows(run["predictions"])
        if mutation == "prediction_count":
            rows = rows[:1]
        elif mutation == "prediction_id":
            rows[1]["id"] = "999"
        else:
            rows[0]["checkpoint_id"] = "another-checkpoint"
        _csv(run["predictions"], rows)
    elif mutation == "summary_identity":
        rows = _csv_rows(run["summary"])
        rows[0]["scientific_variant"] = "another-variant"
        _csv(run["summary"], rows)
    elif mutation == "annotations":
        _json(run["annotations"], [{"id": 1, "text": "changed"}, {"id": 2}])
    else:
        path = run["evaluation"] / "environment.json"
        _json(path, {"source_sha256": "b" * 64})
    _reseal(run)
    report = _inspect(run)
    assert report["status"] == "inconsistent", report
    assert report["jobs"][0]["errors"]


def test_changed_probability_is_detected_with_same_ids_counts_and_checkpoint_id(
    artifact_run: dict,
) -> None:
    rows = _csv_rows(artifact_run["predictions"])
    rows[0]["binary_probability"] = "0.99"
    _csv(artifact_run["predictions"], rows)
    report = _inspect(artifact_run)
    assert report["status"] == "inconsistent"
    errors = report["jobs"][0]["errors"]
    assert "Sealed predictions SHA-256 mismatch" in errors
    assert not any(
        "sample IDs" in error or "row count" in error or "run identity mismatch" in error
        for error in errors
    )


def test_missing_identity_columns_remain_missing_evidence(artifact_run: dict) -> None:
    rows = _csv_rows(artifact_run["predictions"])
    for row in rows:
        del row["checkpoint_id"]
    _csv(artifact_run["predictions"], rows)
    _reseal(artifact_run)
    report = _inspect(artifact_run)
    assert report["status"] == "missing", report
    assert "prediction identity column: checkpoint_id" in report["jobs"][0]["missing"]


def test_missing_manifest_is_a_missing_plan(tmp_path: Path) -> None:
    report = inspect_study_evidence(tmp_path / "not-there.json")
    assert report["status"] == "missing"
    assert report["counts"]["total"] == 0


def test_canonical_hash_preserves_values_and_ignores_mapping_key_order() -> None:
    assert canonical_config_sha256({"a": 1, "b": [2]}) == canonical_config_sha256(
        {"b": [2], "a": 1}
    )
    assert canonical_config_sha256({"a": 1}) != canonical_config_sha256({"a": 2})


@pytest.mark.parametrize(
    "variant,baseline_type", (("b0", "b0_global"), ("b1", "b1_cross_attention"))
)
def test_baseline_without_alignment_loss_matches_not_applicable_metadata(
    artifact_run: dict, variant: str, baseline_type: str
) -> None:
    run = artifact_run
    manifest = json.loads(run["manifest"].read_text(encoding="utf-8"))
    job = manifest["jobs"][0]
    job["variant"] = variant
    actual_train = run["checkpoint"].parent.parent / "effective_config.yaml"
    paths = (
        Path(job["train_config"]),
        Path(job["evaluate_config"]),
        actual_train,
        run["evaluation"] / "effective_config.yaml",
    )
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["project"]["architecture_version"] = "heri_baseline"
        config["model"] = {"baseline_type": baseline_type}
        config["loss"].pop("alignment")
        config["scientific_study"]["variant"] = variant
        _yaml(path, config)
    for phase in ("train", "evaluate"):
        config = yaml.safe_load(Path(job[f"{phase}_config"]).read_text(encoding="utf-8"))
        job[f"{phase}_config_hash"] = canonical_config_sha256(config)
    _json(run["manifest"], manifest)
    metadata_path = checkpoint_metadata_path(run["checkpoint"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["alignment_objective"] = "not_applicable"
    metadata["effective_config_sha256"] = canonical_config_sha256(
        yaml.safe_load(actual_train.read_text(encoding="utf-8"))
    )
    metadata["effective_config"] = yaml.safe_load(actual_train.read_text(encoding="utf-8"))
    _json(metadata_path, metadata)
    for path in (run["summary"], run["predictions"]):
        rows = _csv_rows(path)
        for row in rows:
            row["alignment_objective"] = "not_applicable"
            row["scientific_variant"] = variant
        _csv(path, rows)
    _reseal(run)
    report = _inspect(run)
    assert report["status"] == "internally_consistent", report


def test_source_mismatch_is_rejected_without_an_expected_manifest_source(
    artifact_run: dict,
) -> None:
    run = artifact_run
    _json(run["evaluation"] / "environment.json", {"source_sha256": "b" * 64})
    _reseal(run)
    manifest = json.loads(run["manifest"].read_text(encoding="utf-8"))
    report = inspect_run_evidence(
        manifest["jobs"][0],
        protocol_id=manifest["protocol_id"],
        evaluation_protocol=manifest["evaluation_protocol"],
        expected_test_samples=2,
        base_dir=run["manifest"].parent,
    )
    assert report["status"] == "inconsistent", report
    assert any("Source identity mismatch" in error for error in report["errors"])


def test_resumed_run_config_retains_unchanged_best_checkpoint_snapshot(artifact_run: dict) -> None:
    path = artifact_run["checkpoint"].parent.parent / "effective_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["training"]["resume_from"] = str(path.parent / "checkpoints/last.pt")
    _yaml(path, config)
    _reseal(artifact_run)
    report = _inspect(artifact_run)
    assert report["status"] == "internally_consistent", report


def test_changed_training_horizon_fails_checkpoint_snapshot_comparison(artifact_run: dict) -> None:
    path = artifact_run["checkpoint"].parent.parent / "effective_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["training"]["epochs"] = 21
    _yaml(path, config)
    _reseal(artifact_run)
    report = _inspect(artifact_run)
    assert report["status"] == "inconsistent", report
    assert any("checkpoint-bound training" in error for error in report["jobs"][0]["errors"])


def test_metadata_without_checkpoint_bound_config_snapshot_is_missing_evidence(
    artifact_run: dict,
) -> None:
    path = checkpoint_metadata_path(artifact_run["checkpoint"])
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["effective_config"]
    _json(path, metadata)
    _reseal(artifact_run)
    report = _inspect(artifact_run)
    assert report["status"] == "missing", report
    assert "Checkpoint effective configuration snapshot" in report["jobs"][0]["missing"]


def test_changed_observed_training_data_fails_even_after_resealing(artifact_run: dict) -> None:
    path = artifact_run["checkpoint"].parent.parent / "effective_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["observed_dataset_hashes"]["train"] = "another-observed-train-hash"
    _yaml(path, config)
    _reseal(artifact_run)
    report = _inspect(artifact_run)
    assert report["status"] == "inconsistent", report
    assert any(
        "Observed train annotation SHA-256 mismatch" in error
        for error in report["jobs"][0]["errors"]
    )


def test_missing_observed_training_identity_is_missing_evidence(artifact_run: dict) -> None:
    path = artifact_run["checkpoint"].parent.parent / "effective_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    del config["observed_dataset_hashes"]["train"]
    _yaml(path, config)
    _reseal(artifact_run)
    report = _inspect(artifact_run)
    assert report["status"] == "missing", report
    assert "run observed train annotation SHA-256" in report["jobs"][0]["missing"]
