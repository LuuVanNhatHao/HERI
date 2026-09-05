"""Production orchestration for HERA's controlled cycle-depth study."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.utils.config import ConfigError, dump_yaml, load_yaml_config, validate_hera_config

from .cycle_study import (
    HERA_CYCLE_STUDY_VARIANTS,
    apply_hera_cycle_study_variant,
    cycle_study_protocol,
)


@dataclass(frozen=True)
class CycleStudyJob:
    """Materialized train/evaluation paths for one controlled cycle protocol."""

    variant: str
    train_config_path: Path
    evaluate_config_path: Path
    train_output_root: Path
    evaluate_output_root: Path
    expected_summary_path: Path
    protocol_id: str


def load_cycle_study_suite(path: str | Path) -> dict[str, Any]:
    """Load and validate a cycle-study manifest without loading any model."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ConfigError(f"Cycle-study suite does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("suite"), dict):
        raise ConfigError("Cycle-study manifest must contain a top-level suite mapping")
    suite = dict(manifest["suite"])
    for key in ("name", "base_train_config", "base_evaluate_config", "output_root", "variants"):
        if key not in suite:
            raise ConfigError(f"Missing mandatory cycle-study key: suite.{key}")
    variants = suite["variants"]
    if not isinstance(variants, list) or not variants:
        raise ConfigError("suite.variants must be a non-empty list")
    if len(variants) != len(set(variants)):
        raise ConfigError("suite.variants must not contain duplicates")
    unknown = [name for name in variants if name not in HERA_CYCLE_STUDY_VARIANTS]
    if unknown:
        raise ConfigError(
            f"Unknown cycle-study variants {unknown}; allowed variants: "
            f"{list(HERA_CYCLE_STUDY_VARIANTS)}"
        )
    return suite


def materialize_cycle_study_jobs(
    suite: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> list[CycleStudyJob]:
    """Write complete matched train/evaluation YAMLs for all manifest variants."""

    root = Path(repo_root).resolve()
    train_base = load_yaml_config(_resolve_path(root, suite["base_train_config"]))
    evaluate_base = load_yaml_config(_resolve_path(root, suite["base_evaluate_config"]))
    _validate_canonical_bases(train_base, evaluate_base)
    epochs = int(suite.get("training_epochs", train_base["training"]["epochs"]))
    if epochs <= 0:
        raise ConfigError("suite.training_epochs must be positive")
    micro_batch_size = int(
        suite.get("micro_batch_size", train_base["training"]["micro_batch_size"])
    )
    accumulation = int(
        suite.get(
            "gradient_accumulation_steps",
            train_base["training"]["gradient_accumulation_steps"],
        )
    )
    if micro_batch_size <= 0 or accumulation <= 0:
        raise ConfigError(
            "suite micro_batch_size and gradient_accumulation_steps must be positive"
        )
    protocol_id = _matched_protocol_id(
        train_base,
        training_epochs=epochs,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=accumulation,
    )

    output_root = _resolve_path(root, suite["output_root"])
    jobs: list[CycleStudyJob] = []
    for variant in [str(value) for value in suite["variants"]]:
        variant_root = output_root / variant
        config_root = variant_root / "configs"
        train_root = variant_root / "train"
        evaluate_root = variant_root / "evaluation"
        train_config = apply_hera_cycle_study_variant(train_base, variant)
        evaluate_config = apply_hera_cycle_study_variant(evaluate_base, variant)
        _bind_protocol(
            train_config,
            evaluate_config,
            variant=variant,
            train_root=_portable_path(train_root, root),
            evaluate_root=_portable_path(evaluate_root, root),
            training_epochs=epochs,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=accumulation,
            protocol_id=protocol_id,
        )
        validate_hera_config(train_config, mode="train")
        validate_hera_config(evaluate_config, mode="evaluate")
        config_root.mkdir(parents=True, exist_ok=True)
        train_path = config_root / "train.yaml"
        evaluate_path = config_root / "evaluate.yaml"
        dump_yaml(train_config, train_path)
        dump_yaml(evaluate_config, evaluate_path)
        jobs.append(
            CycleStudyJob(
                variant=variant,
                train_config_path=train_path,
                evaluate_config_path=evaluate_path,
                train_output_root=train_root,
                evaluate_output_root=evaluate_root,
                expected_summary_path=(
                    evaluate_root / str(evaluate_config["project"]["run_id"]) / "summary.csv"
                ),
                protocol_id=protocol_id,
            )
        )
    return jobs


def select_cycle_study_jobs(
    jobs: Sequence[CycleStudyJob],
    selected_variants: Sequence[str] | None,
) -> list[CycleStudyJob]:
    """Select jobs while preserving the study manifest's scientific order."""

    if not selected_variants:
        return list(jobs)
    requested = {str(value) for value in selected_variants}
    configured = {job.variant for job in jobs}
    unknown = sorted(requested - configured)
    if unknown:
        raise ConfigError(f"Requested variants are not enabled by the suite: {unknown}")
    return [job for job in jobs if job.variant in requested]


def run_cycle_study_jobs(
    jobs: Sequence[CycleStudyJob],
    *,
    repo_root: str | Path,
    phase: str = "all",
    environment: Mapping[str, Any] | None = None,
    fail_fast: bool = True,
    prune_last_after_evaluation: bool = False,
) -> list[Path]:
    """Run isolated jobs sequentially with restart-safe completion checks."""

    if phase not in {"all", "train", "evaluate"}:
        raise ValueError("phase must be all, train, or evaluate")
    root = Path(repo_root).resolve()
    process_environment = os.environ.copy()
    process_environment.update(
        {str(key): str(value) for key, value in dict(environment or {}).items()}
    )
    failures: list[str] = []
    for job in jobs:
        if phase == "all" and job.expected_summary_path.is_file():
            print(
                f"[{job.variant}] train/evaluate: final summary already exists; skipping",
                flush=True,
            )
            if prune_last_after_evaluation:
                _prune_completed_last_checkpoints(job)
            continue
        phases: list[tuple[str, Path]] = []
        if phase in {"all", "train"}:
            if _training_is_complete(job):
                print(f"[{job.variant}] train: already complete; skipping", flush=True)
            else:
                phases.append(("train", job.train_config_path))
        if phase in {"all", "evaluate"}:
            if job.expected_summary_path.is_file():
                print(f"[{job.variant}] evaluate: summary already exists; skipping", flush=True)
            else:
                phases.append(("evaluate", job.evaluate_config_path))
        for phase_name, config_path in phases:
            command = [sys.executable, str(root / "scripts" / f"{phase_name}.py"), str(config_path)]
            print(f"[{job.variant}] {phase_name}: {' '.join(command)}", flush=True)
            completed = subprocess.run(command, cwd=root, env=process_environment, check=False)
            complete = completed.returncode == 0
            if phase_name == "train" and complete:
                complete = _training_is_complete(job)
            if complete:
                continue
            failure = (
                f"variant={job.variant} phase={phase_name} exited with code "
                f"{completed.returncode} or did not write its completion artifact"
            )
            failures.append(failure)
            if fail_fast:
                raise RuntimeError(failure)
            break
        if (
            prune_last_after_evaluation
            and phase in {"all", "evaluate"}
            and job.expected_summary_path.is_file()
        ):
            _prune_completed_last_checkpoints(job)
    if failures:
        raise RuntimeError("Cycle-study failures: " + "; ".join(failures))
    return [job.expected_summary_path for job in jobs if job.expected_summary_path.is_file()]


def aggregate_cycle_study_summaries(
    jobs: Iterable[CycleStudyJob],
    *,
    output_path: str | Path,
) -> Path | None:
    """Write the compact cycle-study result source with deltas from single pass."""

    job_list = list(jobs)
    source_rows: list[tuple[CycleStudyJob, dict[str, str]]] = []
    for job in job_list:
        if not job.expected_summary_path.is_file():
            continue
        with job.expected_summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            raise ValueError(f"Expected one row in {job.expected_summary_path}, got {len(rows)}")
        reported = rows[0].get("cycle_study_variant")
        if reported != job.variant:
            raise ValueError(
                f"Cycle-study summary mismatch for {job.variant}: reported {reported!r}"
            )
        reported_protocol = rows[0].get("cycle_study_protocol_id")
        if reported_protocol != job.protocol_id:
            raise ValueError(
                f"Cycle-study protocol mismatch for {job.variant}: "
                f"reported {reported_protocol!r}, expected {job.protocol_id!r}"
            )
        source_rows.append((job, rows[0]))
    if not source_rows:
        return None

    study_complete = len(source_rows) == len(job_list)
    single_cf1 = next(
        (float(row["cf1"]) for job, row in source_rows if job.variant == "single_pass"),
        None,
    )
    output_rows: list[dict[str, Any]] = []
    for job, row in source_rows:
        protocol = cycle_study_protocol(job.variant)
        cf1 = float(row["cf1"])
        output_rows.append(
            {
                "configuration": job.variant,
                "cycle_study_protocol_id": job.protocol_id,
                "study_complete": study_complete,
                "completed_variants": len(source_rows),
                "expected_variants": len(job_list),
                "cycles": protocol["cycles"],
                "rebuild_mode": protocol["rebuild_mode"],
                "acc": row.get("acc", ""),
                "cf1": row.get("cf1", ""),
                "auc": row.get("auc", ""),
                "map": row.get("map", ""),
                "delta_cf1_vs_single_pass": "" if single_cf1 is None else cf1 - single_cf1,
                "delta_cf1_points_vs_single_pass": (
                    "" if single_cf1 is None else 100.0 * (cf1 - single_cf1)
                ),
                "seed": row.get("seed", ""),
                "checkpoint": row.get("checkpoint", ""),
                "summary_path": job.expected_summary_path.as_posix(),
            }
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return destination


def _training_is_complete(job: CycleStudyJob) -> bool:
    if not job.train_output_root.is_dir():
        return False
    markers = sorted(
        job.train_output_root.glob("*/training_complete.json"),
        key=lambda path: (path.stat().st_mtime, path.as_posix()),
        reverse=True,
    )
    for marker in markers:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete":
            continue
        if payload.get("cycle_study_variant") != job.variant:
            continue
        if payload.get("cycle_study_protocol_id") != job.protocol_id:
            continue
        if (marker.parent / "checkpoints" / "best.pt").is_file():
            return True
    return False


def _prune_completed_last_checkpoints(job: CycleStudyJob) -> None:
    """Remove only resumable ``last.pt`` files after a final summary exists."""

    if not job.expected_summary_path.is_file() or not job.train_output_root.is_dir():
        return
    for marker in job.train_output_root.glob("*/training_complete.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "complete":
            continue
        if payload.get("cycle_study_variant") != job.variant:
            continue
        if payload.get("cycle_study_protocol_id") != job.protocol_id:
            continue
        best = marker.parent / "checkpoints" / "best.pt"
        last = marker.parent / "checkpoints" / "last.pt"
        if not best.is_file() or not last.is_file():
            continue
        try:
            last.unlink()
        except OSError as exc:
            print(
                f"[{job.variant}] warning: could not prune {last}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(f"[{job.variant}] pruned resumable checkpoint: {last}", flush=True)


def _validate_canonical_bases(train: Mapping[str, Any], evaluate: Mapping[str, Any]) -> None:
    for label, config, mode in (("training", train, "train"), ("evaluation", evaluate, "evaluate")):
        if config.get("model", {}).get("baseline_type") is not None:
            raise ConfigError(f"The {label} base is B0/B1; cycle studies require HERA")
        if config.get("ablation", {}).get("enabled", False):
            raise ConfigError(f"The {label} base cannot be a component ablation")
        if config.get("cycle_study", {}).get("enabled", False):
            raise ConfigError(f"The {label} base must be canonical, not a prior cycle study")
        validate_hera_config(config, mode=mode)
    if int(train["project"]["seed"]) != int(evaluate["project"]["seed"]):
        raise ConfigError("Train/evaluation bases must use the same project.seed")
    if int(train["dataloader"]["seed"]) != int(evaluate["dataloader"]["seed"]):
        raise ConfigError("Train/evaluation bases must use the same dataloader.seed")


def _bind_protocol(
    train: dict[str, Any],
    evaluate: dict[str, Any],
    *,
    variant: str,
    train_root: str,
    evaluate_root: str,
    training_epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    protocol_id: str,
) -> None:
    train["cycle_study"]["protocol_id"] = protocol_id
    evaluate["cycle_study"]["protocol_id"] = protocol_id
    train["project"]["name"] = f"HERA-DGM4-cycle-{variant}"
    train["project"]["config_name"] = f"cycle_study/{variant}/train"
    train["project"].pop("run_id", None)
    train["output"]["root"] = train_root
    train["training"]["epochs"] = int(training_epochs)
    train["training"]["micro_batch_size"] = int(micro_batch_size)
    train["training"]["gradient_accumulation_steps"] = int(
        gradient_accumulation_steps
    )
    train["dataloader"]["batch_size"] = int(micro_batch_size)
    train["training"]["resume_from"] = None
    train["training"]["auto_resume"]["enabled"] = True
    train["training"]["auto_resume"]["run_dir"] = None

    evaluate["project"]["name"] = f"HERA-DGM4-cycle-{variant}"
    evaluate["project"]["config_name"] = f"cycle_study/{variant}/evaluate"
    evaluate["project"]["run_id"] = f"hera_cycle_{variant}_test"
    evaluate["output"]["root"] = evaluate_root
    evaluate["training"]["epochs"] = int(training_epochs)
    evaluate["evaluation"]["checkpoint_path"] = None
    evaluate["evaluation"]["thresholds_path"] = None
    resolution = evaluate["evaluation"]["checkpoint_resolution"]
    resolution["enabled"] = True
    resolution["mode"] = "latest"
    resolution["output_root"] = train_root
    resolution["run_dir"] = None


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _matched_protocol_id(
    train_base: Mapping[str, Any],
    *,
    training_epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> str:
    """Fingerprint every matched training field except the controlled cycle knobs."""

    payload = copy.deepcopy(dict(train_base))
    payload.pop("output", None)
    project = payload.get("project", {})
    for key in ("name", "config_name", "run_id"):
        project.pop(key, None)
    training = payload["training"]
    training["epochs"] = int(training_epochs)
    training["micro_batch_size"] = int(micro_batch_size)
    training["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    training["resume_from"] = None
    training["auto_resume"]["run_dir"] = None
    payload["dataloader"]["batch_size"] = int(micro_batch_size)
    reasoning = payload["graph"]["reasoning"]
    reasoning["cycles"] = "CONTROLLED_CYCLE_DEPTH"
    reasoning["dynamic_rebuild"] = "CONTROLLED_REBUILD_MODE"
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
