from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from tests.support.paths import REPO_ROOT

import pytest

from heri.data.samm import SAMM_EXPECTED_SHA256
from heri.evaluation.protocol import EVALUATION_PROTOCOL
from heri.experiments.ablation import get_hera_ablation_variant
from heri.experiments.samm_matched_multiseed import (
    SAMM_MATCHED_VARIANTS,
    _prune_job_checkpoints,
    _training_is_complete,
    _validate_prediction_csv,
    aggregate_samm_matched_results,
    load_samm_matched_suite,
    materialize_samm_matched_jobs,
    run_samm_matched_jobs,
    select_samm_matched_jobs,
)
from heri.training.state import TrainingState
from heri.training.trainer import BaselineTrainer
from heri.utils.config import (
    ConfigError,
    dump_yaml,
    load_yaml_config,
    validate_baseline_config,
    validate_hera_config,
)


ROOT = REPO_ROOT
SUITE_PATH = ROOT / "config" / "experiments" / "samm" / "matched_multiseed.yaml"


def _write_summary(
    path: Path,
    *,
    variant: str,
    seed: int,
    class_macro_f1: float = 0.90,
    legacy_study: bool = False,
) -> None:
    row = {
        "dataset_name": "SAMM",
        "split": "test",
        "sample_count": 50_000,
        "seed": seed,
        "checkpoint_evaluation_protocol": EVALUATION_PROTOCOL,
        "dataset_hashes": json.dumps(SAMM_EXPECTED_SHA256),
        "scientific_study_id": "" if legacy_study else "samm_heri_grounded_fixed05",
        "scientific_variant": "" if legacy_study else variant,
        "baseline_type": "b1_cross_attention" if variant == "b1" else "",
        "model_variant": (
            "" if variant == "b1" else "hera" if variant == "hera" else f"hera_{variant}"
        ),
        "ablation_variant": (
            ""
            if variant == "b1"
            else "canonical"
            if variant in {"hera", "two_cycle_static"}
            else variant
        ),
        "text_grounding_aggregation": ("generic_head" if variant == "b1" else "relative_soft"),
        "image_presence_aggregation": "atomic_any_fixed_0.5",
        "image_presence_threshold": 0.5,
        "cycle_study_variant": "two_cycle_static" if variant == "two_cycle_static" else "none",
        "decision_threshold_source": "fixed_0.5",
        "evidence_sufficiency_protocol": ("" if variant == "b1" else "per_modality_v1"),
        "evidence_sufficiency_aggregation": ("" if variant == "b1" else "per_modality"),
        "evidence_sufficiency_validity": (
            "" if variant == "b1" else "positive_hypothesis_and_valid_annotation"
        ),
        "samm_released_all_sample_image_count": 50_000,
        "samm_official_map": class_macro_f1 + 0.02,
        "samm_official_cf1": class_macro_f1,
        "samm_official_of1": class_macro_f1 + 0.01,
        "class_macro_f1": class_macro_f1,
        "class_accuracy": class_macro_f1 - 0.01,
        "acc": class_macro_f1 - 0.02,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_predictions(path: Path, count: int = 50_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "binary_probability",
                "image_presence_probability",
                "predicted_image_presence",
            ]
        )
        writer.writerows((index, 0.5, 0.5, 1) for index in range(count))


def test_manifest_prespecifies_four_variants_and_three_seeds() -> None:
    suite = load_samm_matched_suite(SUITE_PATH)

    assert tuple(suite["variants"]) == SAMM_MATCHED_VARIANTS
    assert suite["seeds"] == [7, 21, 101]
    assert suite["expected_test_samples"] == 50_000
    assert suite["retention"] == {
        "minimum_free_gib_before_train": 55,
        "delete_last_after_evaluation": True,
        "delete_best_after_evaluation": False,
    }


def test_materializes_twelve_isolated_matched_jobs(tmp_path: Path) -> None:
    suite = load_samm_matched_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "matched")
    jobs = materialize_samm_matched_jobs(suite, repo_root=ROOT)

    assert [(job.seed, job.variant) for job in jobs] == [
        (seed, variant) for seed in (7, 21, 101) for variant in SAMM_MATCHED_VARIANTS
    ]
    assert len({job.train_run_dir for job in jobs}) == 15
    assert len({job.evaluation_run_dir for job in jobs}) == 15

    for job in jobs:
        train = load_yaml_config(job.train_config_path)
        evaluate = load_yaml_config(job.evaluate_config_path)
        validator = validate_baseline_config if job.variant == "b1" else validate_hera_config
        validator(train, mode="train")
        validator(evaluate, mode="evaluate")
        assert train["dataset_hashes"] == SAMM_EXPECTED_SHA256
        assert train["project"]["seed"] == job.seed
        assert train["dataloader"]["seed"] == job.seed
        assert evaluate["project"]["seed"] == job.seed
        assert evaluate["dataloader"]["seed"] == job.seed
        assert train["training"]["epochs"] == 20
        assert train["training"]["stages"]["stage_a_freeze_epochs"] == 3
        assert train["training"]["stages"]["stage_b_full_unfreeze"]["start_epoch"] == 4
        assert train["training"]["early_stopping"]["patience"] == 3
        assert not train["training"]["early_stopping"]["enabled"]
        assert train["training"]["early_stopping"]["metric"] == "samm_official_cf1"
        assert not train["thresholding"]["enabled"]
        assert train["checkpointing"]["selection_metric"] == "samm_official_cf1"
        assert train["scientific_study"] == {
            "protocol_id": "samm_heri_grounded_fixed05",
            "variant": job.variant,
            "seed": job.seed,
        }
        assert (
            Path(evaluate["evaluation"]["checkpoint_path"]).resolve()
            == (job.train_run_dir / "checkpoints/best.pt").resolve()
        )
        assert (
            Path(evaluate["evaluation"]["thresholds_path"]).resolve()
            == (job.train_run_dir / "thresholds/best_thresholds.json").resolve()
        )
        if job.variant == "b1":
            assert train["model"]["baseline_type"] == "b1_cross_attention"
        else:
            expected = None if job.variant in {"hera", "two_cycle_static"} else job.variant
            assert get_hera_ablation_variant(train) == expected
            assert evaluate["model"]["grounding"]["text"]["aggregation"] == "relative_soft"


def test_selection_preserves_manifest_order_and_rejects_unknown(tmp_path: Path) -> None:
    suite = load_samm_matched_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "matched")
    jobs = materialize_samm_matched_jobs(suite, repo_root=ROOT)

    selected = select_samm_matched_jobs(
        jobs,
        variants=["generic_grounding", "hera"],
        seeds=[21, 7],
    )
    assert [(job.seed, job.variant) for job in selected] == [
        (7, "hera"),
        (7, "generic_grounding"),
        (21, "hera"),
        (21, "generic_grounding"),
    ]
    with pytest.raises(ConfigError, match="Unknown SAMM matched variants"):
        select_samm_matched_jobs(jobs, variants=["gatv2"])
    with pytest.raises(ConfigError, match="absent from the suite"):
        select_samm_matched_jobs(jobs, seeds=[999])


def test_adopts_only_full_seed42_artifact_pair_without_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_samm_matched_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "matched")
    original = materialize_samm_matched_jobs(suite, repo_root=ROOT)[0]
    source = tmp_path / "standalone" / "summary.csv"
    _write_summary(source, variant="hera", seed=7, legacy_study=True)
    _write_predictions(source.parent / "predictions" / "test_predictions.csv")
    job = replace(original, reusable_summary_path=source)

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("Validated reusable seed-42 results must skip subprocesses")

    monkeypatch.setattr(
        "heri.experiments.samm_matched_multiseed.subprocess.run",
        fail_if_launched,
    )
    summaries = run_samm_matched_jobs(
        [job],
        suite=suite,
        repo_root=ROOT,
        phase="all",
    )

    assert summaries == [job.expected_summary_path]
    assert job.expected_summary_path.is_file()
    assert (job.evaluation_run_dir / "predictions" / "test_predictions.csv").is_file()
    marker = json.loads(
        (job.evaluation_run_dir / "adopted_summary.json").read_text(encoding="utf-8")
    )
    assert marker["source"] == str(source)


def test_prediction_validation_rejects_truncation(tmp_path: Path) -> None:
    predictions = tmp_path / "test_predictions.csv"
    _write_predictions(predictions, count=2)

    _validate_prediction_csv(predictions, expected_samples=2)
    with pytest.raises(ValueError, match="has 2 rows; expected 3"):
        _validate_prediction_csv(predictions, expected_samples=3)


def test_training_completion_is_bound_to_suite_protocol(tmp_path: Path) -> None:
    suite = load_samm_matched_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "matched")
    job = materialize_samm_matched_jobs(suite, repo_root=ROOT)[0]
    (job.train_run_dir / "checkpoints").mkdir(parents=True)
    (job.train_run_dir / "thresholds").mkdir(parents=True)
    (job.train_run_dir / "checkpoints" / "best.pt").write_bytes(b"best")
    (job.train_run_dir / "thresholds" / "best_thresholds.json").write_text("{}", encoding="utf-8")
    (job.train_run_dir / "training_complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    effective = load_yaml_config(job.train_config_path)
    dump_yaml(effective, job.train_run_dir / "effective_config.yaml")

    assert _training_is_complete(job, suite=suite)
    effective["scientific_study"]["protocol_id"] = "wrong"
    dump_yaml(effective, job.train_run_dir / "effective_config.yaml")
    assert not _training_is_complete(job, suite=suite)


def test_valid_evaluation_prunes_only_suite_best_and_last(tmp_path: Path) -> None:
    suite = load_samm_matched_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "matched")
    job = materialize_samm_matched_jobs(suite, repo_root=ROOT)[0]
    checkpoints = job.train_run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "best.pt").write_bytes(b"best")
    (checkpoints / "last.pt").write_bytes(b"last")
    (checkpoints / "keep.txt").write_text("keep", encoding="utf-8")

    _prune_job_checkpoints(job, suite=suite)

    assert (checkpoints / "best.pt").read_bytes() == b"best"
    assert not (checkpoints / "last.pt").exists()
    assert (checkpoints / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_aggregation_reports_sample_std_and_paired_deltas(tmp_path: Path) -> None:
    suite = load_samm_matched_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "matched")
    jobs = materialize_samm_matched_jobs(suite, repo_root=ROOT)
    bases = {
        "hera": 0.95,
        "b1": 0.90,
        "two_cycle_static": 0.94,
        "no_router": 0.92,
        "generic_grounding": 0.91,
    }
    seed_offsets = {7: 0.00, 21: 0.01, 101: -0.01}
    for job in jobs:
        _write_summary(
            job.expected_summary_path,
            variant=job.variant,
            seed=job.seed,
            class_macro_f1=bases[job.variant] + seed_offsets[job.seed],
        )

    artifacts = aggregate_samm_matched_results(
        jobs,
        suite=suite,
        output_root=tmp_path / "aggregate",
    )

    assert artifacts is not None
    with artifacts["variant_statistics"].open("r", encoding="utf-8", newline="") as handle:
        rows = {(row["variant"], row["metric"]): row for row in csv.DictReader(handle)}
    hera = rows[("hera", "class_macro_f1")]
    assert hera["n"] == "3"
    assert hera["complete"] == "True"
    assert math.isclose(float(hera["mean"]), 0.95)
    assert math.isclose(float(hera["std"]), 0.01)

    with artifacts["paired_statistics"].open("r", encoding="utf-8", newline="") as handle:
        paired = {(row["variant"], row["metric"]): row for row in csv.DictReader(handle)}
    b1_delta = paired[("b1", "class_macro_f1")]
    assert b1_delta["n"] == "3"
    assert math.isclose(float(b1_delta["mean"]), 0.05)
    assert math.isclose(float(b1_delta["std"]), 0.0, abs_tol=1e-12)


def test_baseline_completion_marker_and_completed_horizon_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_yaml_config(
        ROOT / "config" / "datasets" / "samm" / "baselines" / "b1" / "train.yaml"
    )
    run_dir = tmp_path / "b1_run"
    best = run_dir / "checkpoints" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"best")
    checkpoint = {
        "config": config,
        "epoch": int(config["training"]["epochs"]),
        "best_checkpoint": str(best),
    }
    trainer = BaselineTrainer.__new__(BaselineTrainer)
    trainer.config = config
    trainer.train_loader = [object()]
    trainer.resume_checkpoint_path = run_dir / "checkpoints" / "last.pt"
    trainer.run_dir = run_dir
    trainer.state = TrainingState()
    monkeypatch.setattr(
        "heri.training.trainer._load_hera_resume_payload",
        lambda path: (checkpoint, path),
    )

    returned = trainer.run()

    assert returned == best
    marker = json.loads((run_dir / "training_complete.json").read_text(encoding="utf-8"))
    assert marker == {
        "status": "complete",
        "final_epoch": 20,
        "stop_reason": "max_epochs",
        "best_checkpoint": str(best),
        "baseline_type": "b1_cross_attention",
    }
