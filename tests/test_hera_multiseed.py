from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments.multiseed_runner import (
    aggregate_multiseed_summaries,
    load_multiseed_suite,
    materialize_multiseed_jobs,
    run_multiseed_jobs,
    select_multiseed_jobs,
)
from src.utils.config import (
    ConfigError,
    dump_yaml,
    load_yaml_config,
    validate_hera_config,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "config" / "experiments" / "hera_multiseed.yaml"


def test_manifest_defines_three_unique_canonical_seeds() -> None:
    suite = load_multiseed_suite(SUITE_PATH)
    assert suite["seeds"] == [7, 123, 3407]
    assert len(set(suite["seeds"])) == 3


def test_materialized_jobs_are_isolated_and_protocol_matched(tmp_path: Path) -> None:
    suite = load_multiseed_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "multiseed")
    jobs = materialize_multiseed_jobs(suite, repo_root=ROOT)

    assert [job.seed for job in jobs] == [7, 123, 3407]
    assert len({job.train_output_root for job in jobs}) == 3
    for job in jobs:
        train = load_yaml_config(job.train_config_path)
        evaluate = load_yaml_config(job.evaluate_config_path)
        validate_hera_config(train, mode="train")
        validate_hera_config(evaluate, mode="evaluate")
        assert train["model"]["model_type"] == "hera"
        assert train["model"].get("baseline_type") is None
        assert not train.get("ablation", {}).get("enabled", False)
        assert train["project"]["seed"] == job.seed
        assert train["dataloader"]["seed"] == job.seed
        assert evaluate["project"]["seed"] == job.seed
        assert evaluate["dataloader"]["seed"] == job.seed
        assert train["training"]["epochs"] == 20
        assert train["training"]["stages"]["stage_a_freeze_epochs"] == 3
        assert train["training"]["stages"]["stage_b_full_unfreeze"] == {
            "enabled": True,
            "start_epoch": 4,
        }
        assert train["training"]["micro_batch_size"] == 16
        assert train["dataloader"]["batch_size"] == 16
        assert train["training"]["auto_resume"]["enabled"] is True
        resolution = evaluate["evaluation"]["checkpoint_resolution"]
        assert resolution["mode"] == "latest"
        assert resolution["output_root"] == train["output"]["root"]
        assert resolution["checkpoint_name"] == "best.pt"


def test_training_epoch_override_cannot_drift_from_canonical_base(
    tmp_path: Path,
) -> None:
    suite = load_multiseed_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "multiseed")
    suite["training_epochs"] = 12
    with pytest.raises(ConfigError, match="canonical base training horizon"):
        materialize_multiseed_jobs(suite, repo_root=ROOT)


def test_seed_subset_preserves_manifest_order_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    suite = load_multiseed_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "multiseed")
    jobs = materialize_multiseed_jobs(suite, repo_root=ROOT)
    assert [job.seed for job in select_multiseed_jobs(jobs, [3407, 7])] == [7, 3407]
    with pytest.raises(ConfigError, match="not enabled"):
        select_multiseed_jobs(jobs, [999])


def test_existing_summary_skips_retraining_after_checkpoint_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_multiseed_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "multiseed")
    job = materialize_multiseed_jobs(suite, repo_root=ROOT)[0]
    job.expected_summary_path.parent.mkdir(parents=True, exist_ok=True)
    job.expected_summary_path.write_text("seed,acc\n7,0.91\n", encoding="utf-8")

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("A completed seed must not launch a subprocess")

    monkeypatch.setattr(
        "src.experiments.multiseed_runner.subprocess.run", fail_if_launched
    )
    summaries = run_multiseed_jobs([job], repo_root=ROOT, phase="all")
    assert summaries == [job.expected_summary_path]


def test_train_completion_requires_matching_seed_and_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_multiseed_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "multiseed")
    job = materialize_multiseed_jobs(suite, repo_root=ROOT)[0]
    run_dir = job.train_output_root / "hera_seed7_fixture"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"fixture")
    (run_dir / "training_complete.json").write_text(
        json.dumps({"status": "complete", "ablation_variant": None}),
        encoding="utf-8",
    )
    effective = load_yaml_config(job.train_config_path)
    dump_yaml(effective, run_dir / "effective_config.yaml")

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("Matching completed training must be skipped")

    monkeypatch.setattr(
        "src.experiments.multiseed_runner.subprocess.run", fail_if_launched
    )
    run_multiseed_jobs([job], repo_root=ROOT, phase="train")

    effective["project"]["seed"] = 123
    dump_yaml(effective, run_dir / "effective_config.yaml")
    monkeypatch.setattr(
        "src.experiments.multiseed_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(RuntimeError, match="phase=train"):
        run_multiseed_jobs([job], repo_root=ROOT, phase="train")


def test_aggregation_validates_seed_and_computes_sample_std(tmp_path: Path) -> None:
    suite = load_multiseed_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "multiseed")
    jobs = materialize_multiseed_jobs(suite, repo_root=ROOT)
    values = {7: 0.90, 123: 0.91, 3407: 0.92}
    for job in jobs:
        job.expected_summary_path.parent.mkdir(parents=True, exist_ok=True)
        job.expected_summary_path.write_text(
            f"seed,acc,cf1\n{job.seed},{values[job.seed]},0.88\n",
            encoding="utf-8",
        )

    combined, stats = aggregate_multiseed_summaries(
        jobs,
        output_root=tmp_path / "aggregate",
        metrics=["acc", "cf1"],
    )
    assert combined is not None and stats is not None
    with stats.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert int(rows["acc"]["n"]) == 3
    assert rows["acc"]["complete"] == "True"
    assert math.isclose(float(rows["acc"]["mean"]), 0.91)
    assert math.isclose(float(rows["acc"]["std"]), 0.01)

    jobs[0].expected_summary_path.write_text("seed,acc\n999,0.90\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed mismatch"):
        aggregate_multiseed_summaries(
            jobs,
            output_root=tmp_path / "aggregate",
            metrics=["acc"],
        )
