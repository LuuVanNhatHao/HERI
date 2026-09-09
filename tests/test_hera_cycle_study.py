from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.experiments import (
    HERA_CYCLE_STUDY_VARIANTS,
    apply_hera_cycle_study_variant,
    cycle_study_protocol,
    validate_hera_checkpoint_variant,
)
from src.experiments.cycle_study_runner import (
    aggregate_cycle_study_summaries,
    load_cycle_study_suite,
    materialize_cycle_study_jobs,
    run_cycle_study_jobs,
    select_cycle_study_jobs,
)
from src.utils.config import ConfigError, load_yaml_config, validate_hera_config


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "config" / "experiments" / "dgm4" / "cycle_study.yaml"


def test_manifest_defines_complete_controlled_cycle_study() -> None:
    suite = load_cycle_study_suite(SUITE_PATH)
    assert tuple(suite["variants"]) == HERA_CYCLE_STUDY_VARIANTS
    assert cycle_study_protocol("single_pass") == {
        "cycles": 1,
        "dynamic_rebuild": False,
        "rebuild_mode": "none",
    }
    assert cycle_study_protocol("two_cycle_static")["rebuild_mode"] == "static"
    assert cycle_study_protocol("full_hera")["cycles"] == 2
    assert cycle_study_protocol("three_cycle_extension")["cycles"] == 3


@pytest.mark.parametrize("variant", HERA_CYCLE_STUDY_VARIANTS)
def test_cycle_variant_is_valid_but_canonical_protocol_stays_locked(variant: str) -> None:
    canonical = load_yaml_config(ROOT / "config/datasets/dgm4/hera/train.yaml")
    effective = apply_hera_cycle_study_variant(canonical, variant)

    validate_hera_config(effective, mode="train")
    protocol = cycle_study_protocol(variant)
    assert effective["graph"]["reasoning"]["cycles"] == protocol["cycles"]
    assert (
        effective["graph"]["reasoning"]["dynamic_rebuild"]
        is protocol["dynamic_rebuild"]
    )
    assert canonical["graph"]["reasoning"] == {
        "cycles": 2,
        "hypothesis_feedback": True,
        "dynamic_rebuild": True,
        "hypothesis_edge_reweighting": "cosine",
    }


def test_unlabeled_noncanonical_cycle_config_is_rejected() -> None:
    config = load_yaml_config(ROOT / "config/datasets/dgm4/hera/train.yaml")
    config["graph"]["reasoning"]["cycles"] = 3
    with pytest.raises(ConfigError, match="exactly two"):
        validate_hera_config(config, mode="train")


def test_cycle_study_rejects_component_ablation_combination() -> None:
    config = load_yaml_config(ROOT / "config/datasets/dgm4/hera/train.yaml")
    config["ablation"] = {"enabled": True, "variant": "no_router"}
    config["cycle_study"] = {
        "enabled": True,
        "variant": "single_pass",
        "rebuild_mode": "none",
    }
    config["graph"]["reasoning"]["cycles"] = 1
    config["graph"]["reasoning"]["dynamic_rebuild"] = False
    with pytest.raises(ConfigError, match="both ablation and cycle_study"):
        validate_hera_config(config, mode="train")


def test_materialized_jobs_are_isolated_and_resume_safe(tmp_path: Path) -> None:
    suite = load_cycle_study_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "cycles")
    jobs = materialize_cycle_study_jobs(suite, repo_root=ROOT)

    assert [job.variant for job in jobs] == list(HERA_CYCLE_STUDY_VARIANTS)
    assert len({job.train_output_root for job in jobs}) == 4
    for job in jobs:
        train = load_yaml_config(job.train_config_path)
        evaluate = load_yaml_config(job.evaluate_config_path)
        validate_hera_config(train, mode="train")
        validate_hera_config(evaluate, mode="evaluate")
        assert train["training"]["epochs"] == 20
        assert train["training"]["micro_batch_size"] == 16
        assert train["training"]["gradient_accumulation_steps"] == 1
        assert train["dataloader"]["batch_size"] == 16
        assert len(train["cycle_study"]["protocol_id"]) == 64
        assert (
            evaluate["cycle_study"]["protocol_id"]
            == train["cycle_study"]["protocol_id"]
        )
        assert train["training"]["auto_resume"]["enabled"] is True
        assert (
            evaluate["evaluation"]["checkpoint_resolution"]["output_root"]
            == train["output"]["root"]
        )

    assert [job.variant for job in select_cycle_study_jobs(jobs, ["full_hera", "single_pass"])] == [
        "single_pass",
        "full_hera",
    ]
    with pytest.raises(ConfigError, match="not enabled"):
        select_cycle_study_jobs(jobs, ["unknown"])


def test_checkpoint_identity_prevents_cross_cycle_resume() -> None:
    canonical = load_yaml_config(ROOT / "config/datasets/dgm4/hera/train.yaml")
    static = apply_hera_cycle_study_variant(canonical, "two_cycle_static")
    dynamic = apply_hera_cycle_study_variant(canonical, "full_hera")

    validate_hera_checkpoint_variant(static, static, context="fixture")
    with pytest.raises(ValueError, match="cycle-study mismatch"):
        validate_hera_checkpoint_variant(static, dynamic, context="fixture")
    with pytest.raises(ValueError, match="cycle-study mismatch"):
        validate_hera_checkpoint_variant(dynamic, canonical, context="fixture")


def test_summary_aggregation_computes_table_delta(tmp_path: Path) -> None:
    suite = load_cycle_study_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "cycles")
    jobs = materialize_cycle_study_jobs(suite, repo_root=ROOT)
    cf1_values = {
        "single_pass": 0.89401,
        "two_cycle_static": 0.90047,
        "full_hera": 0.90701,
        "three_cycle_extension": 0.90617,
    }
    for job in jobs:
        job.expected_summary_path.parent.mkdir(parents=True, exist_ok=True)
        job.expected_summary_path.write_text(
            "cycle_study_variant,cycle_study_protocol_id,acc,cf1,auc,map,seed,checkpoint\n"
            f"{job.variant},{job.protocol_id},0.9,{cf1_values[job.variant]},"
            "0.95,0.96,42,best.pt\n",
            encoding="utf-8",
        )

    output = aggregate_cycle_study_summaries(
        jobs,
        output_path=tmp_path / "cycle_study_summary.csv",
    )
    assert output is not None
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["configuration"]: row for row in csv.DictReader(handle)}
    assert rows["full_hera"]["study_complete"] == "True"
    assert (
        float(rows["single_pass"]["delta_cf1_points_vs_single_pass"]) == 0.0
    )
    assert float(rows["two_cycle_static"]["delta_cf1_points_vs_single_pass"]) == pytest.approx(0.646)
    assert float(rows["full_hera"]["delta_cf1_points_vs_single_pass"]) == pytest.approx(1.3)


def test_completed_cycle_job_is_idempotently_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_cycle_study_suite(SUITE_PATH)
    suite["output_root"] = str(tmp_path / "cycles")
    job = materialize_cycle_study_jobs(suite, repo_root=ROOT)[0]
    run_dir = job.train_output_root / "hera_seed42_fixture"
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    last = run_dir / "checkpoints" / "last.pt"
    last.write_bytes(b"resume fixture")
    (run_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cycle_study_variant": job.variant,
                "cycle_study_protocol_id": job.protocol_id,
                "best_checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    job.expected_summary_path.parent.mkdir(parents=True, exist_ok=True)
    job.expected_summary_path.write_text(
        "cycle_study_variant,cycle_study_protocol_id,cf1\n"
        f"{job.variant},{job.protocol_id},0.89\n",
        encoding="utf-8",
    )

    def fail_if_launched(*args, **kwargs):
        raise AssertionError("Completed cycle-study phases must not relaunch")

    monkeypatch.setattr(
        "src.experiments.cycle_study_runner.subprocess.run", fail_if_launched
    )
    summaries = run_cycle_study_jobs(
        [job],
        repo_root=ROOT,
        phase="all",
        prune_last_after_evaluation=True,
    )
    assert summaries == [job.expected_summary_path]
    assert checkpoint.is_file()
    assert not last.exists()
