"""Production orchestration for canonical HERA multi-seed experiments.

The runner deliberately delegates model training and evaluation to the same
entry points used by the canonical experiment.  It changes only random seeds,
artifact names, and isolated output roots; the architecture, data, losses,
optimization, staged full-fine-tuning, and threshold protocols remain bound to
the checked-in HERA train/evaluation configs.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from heri.utils.config import (
    ConfigError,
    dump_yaml,
    load_yaml_config,
    validate_hera_config,
)


_SHARED_PROTOCOL_SECTIONS: tuple[str, ...] = (
    "dataset",
    "dataset_hashes",
    "image",
    "tokenizer",
    "backbones",
    "evidence",
    "routing",
    "graph",
    "model",
    "loss",
    "thresholding",
)


@dataclass(frozen=True)
class MultiSeedJob:
    """Materialized production paths for one canonical HERA seed."""

    seed: int
    train_config_path: Path
    evaluate_config_path: Path
    train_output_root: Path
    evaluate_output_root: Path
    expected_summary_path: Path


def load_multiseed_suite(path: str | Path) -> dict[str, Any]:
    """Load and validate a multi-seed manifest without loading a model."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ConfigError(f"Multi-seed suite does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("suite"), dict):
        raise ConfigError("Multi-seed manifest must contain a top-level suite mapping")
    suite = dict(manifest["suite"])
    mandatory = (
        "name",
        "base_train_config",
        "base_evaluate_config",
        "output_root",
        "seeds",
    )
    for key in mandatory:
        if key not in suite:
            raise ConfigError(f"Missing mandatory multi-seed key: suite.{key}")
    seeds = _validated_seeds(suite["seeds"], context="suite.seeds")
    if len(seeds) < 2:
        raise ConfigError("suite.seeds must contain at least two independent seeds")
    metrics = suite.get("aggregate_metrics", [])
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("suite.aggregate_metrics must be a non-empty list")
    if len(metrics) != len(set(str(value) for value in metrics)):
        raise ConfigError("suite.aggregate_metrics must not contain duplicates")
    suite["seeds"] = seeds
    return suite


def materialize_multiseed_jobs(
    suite: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> list[MultiSeedJob]:
    """Write complete per-seed production configs and return their jobs."""

    root = Path(repo_root).resolve()
    seeds = _validated_seeds(suite["seeds"], context="suite.seeds")
    train_base = load_yaml_config(_resolve_path(root, suite["base_train_config"]))
    evaluate_base = load_yaml_config(_resolve_path(root, suite["base_evaluate_config"]))
    _validate_canonical_bases(train_base, evaluate_base)

    configured_epochs = suite.get("training_epochs")
    if configured_epochs is not None and int(configured_epochs) != int(
        train_base["training"]["epochs"]
    ):
        raise ConfigError(
            "suite.training_epochs must equal the canonical base training horizon; "
            "multi-seed runs may change seeds, not the optimization protocol"
        )

    output_root = _resolve_path(root, suite["output_root"])
    jobs: list[MultiSeedJob] = []
    reference_train: dict[str, Any] | None = None
    reference_evaluate: dict[str, Any] | None = None
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        config_root = seed_root / "configs"
        train_root = seed_root / "train"
        evaluate_root = seed_root / "evaluation"
        train_config = copy.deepcopy(train_base)
        evaluate_config = copy.deepcopy(evaluate_base)
        _bind_seed_protocol(
            train_config,
            evaluate_config,
            seed=seed,
            train_root=_portable_path(train_root, root),
            evaluate_root=_portable_path(evaluate_root, root),
        )
        validate_hera_config(train_config, mode="train")
        validate_hera_config(evaluate_config, mode="evaluate")
        _assert_seed_is_the_only_experimental_change(
            train_config,
            reference_train,
            mode="train",
        )
        _assert_seed_is_the_only_experimental_change(
            evaluate_config,
            reference_evaluate,
            mode="evaluate",
        )
        reference_train = reference_train or copy.deepcopy(train_config)
        reference_evaluate = reference_evaluate or copy.deepcopy(evaluate_config)

        train_config_path = config_root / "train.yaml"
        evaluate_config_path = config_root / "evaluate.yaml"
        config_root.mkdir(parents=True, exist_ok=True)
        dump_yaml(train_config, train_config_path)
        dump_yaml(evaluate_config, evaluate_config_path)
        evaluation_run_id = str(evaluate_config["project"]["run_id"])
        jobs.append(
            MultiSeedJob(
                seed=seed,
                train_config_path=train_config_path,
                evaluate_config_path=evaluate_config_path,
                train_output_root=train_root,
                evaluate_output_root=evaluate_root,
                expected_summary_path=evaluate_root / evaluation_run_id / "summary.csv",
            )
        )
    return jobs


def select_multiseed_jobs(
    jobs: Sequence[MultiSeedJob], selected_seeds: Sequence[int] | None
) -> list[MultiSeedJob]:
    """Select manifest jobs while preserving manifest order."""

    if not selected_seeds:
        return list(jobs)
    requested = _validated_seeds(selected_seeds, context="--seeds")
    configured = {job.seed for job in jobs}
    unknown = [seed for seed in requested if seed not in configured]
    if unknown:
        raise ConfigError(
            f"Requested seeds are not enabled by the suite manifest: {unknown}"
        )
    requested_set = set(requested)
    return [job for job in jobs if job.seed in requested_set]


def run_multiseed_jobs(
    jobs: Sequence[MultiSeedJob],
    *,
    repo_root: str | Path,
    phase: str = "all",
    environment: Mapping[str, Any] | None = None,
    fail_fast: bool = True,
) -> list[Path]:
    """Run seeds sequentially with crash-safe resume and idempotent skipping."""

    if phase not in {"all", "train", "evaluate"}:
        raise ValueError("phase must be all, train, or evaluate")
    root = Path(repo_root).resolve()
    process_environment = os.environ.copy()
    for key, value in dict(environment or {}).items():
        process_environment[str(key)] = str(value)

    failures: list[str] = []
    for job in jobs:
        # A persisted test result remains authoritative after intentional
        # checkpoint cleanup.  An `all` rerun must never retrain that seed.
        if phase == "all" and job.expected_summary_path.is_file():
            print(
                f"[seed={job.seed}] complete summary exists; skipping train/evaluate",
                flush=True,
            )
            continue

        if phase in {"all", "train"}:
            if _training_is_complete(job):
                print(
                    f"[seed={job.seed}] train: already complete; skipping", flush=True
                )
            elif not _run_phase(job, "train", root, process_environment):
                failure = f"seed={job.seed} phase=train failed or omitted its completion artifact"
                failures.append(failure)
                if fail_fast:
                    raise RuntimeError(failure)
                continue

        if phase in {"all", "evaluate"}:
            if job.expected_summary_path.is_file():
                print(
                    f"[seed={job.seed}] evaluate: summary already exists; skipping",
                    flush=True,
                )
            else:
                if not _training_is_complete(job):
                    failure = (
                        f"seed={job.seed} phase=evaluate has no complete matching training run; "
                        "run the training phase first"
                    )
                    failures.append(failure)
                    if fail_fast:
                        raise RuntimeError(failure)
                    continue
                if not _run_phase(job, "evaluate", root, process_environment):
                    failure = (
                        f"seed={job.seed} phase=evaluate failed or omitted summary.csv"
                    )
                    failures.append(failure)
                    if fail_fast:
                        raise RuntimeError(failure)

    if failures:
        raise RuntimeError("Multi-seed suite failures: " + "; ".join(failures))
    return [
        job.expected_summary_path for job in jobs if job.expected_summary_path.is_file()
    ]


def aggregate_multiseed_summaries(
    jobs: Iterable[MultiSeedJob],
    *,
    output_root: str | Path,
    metrics: Sequence[str],
) -> tuple[Path | None, Path | None]:
    """Write per-seed rows and paper-ready mean/sample-standard-deviation rows."""

    job_list = list(jobs)
    rows: list[dict[str, Any]] = []
    field_order: list[str] = ["seed"]
    for job in job_list:
        if not job.expected_summary_path.is_file():
            continue
        with job.expected_summary_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            summary_rows = list(csv.DictReader(handle))
        if len(summary_rows) != 1:
            raise ValueError(
                f"Expected one summary row in {job.expected_summary_path}, got {len(summary_rows)}"
            )
        reported_seed = int(summary_rows[0]["seed"])
        if reported_seed != job.seed:
            raise ValueError(
                f"Summary seed mismatch for configured seed {job.seed}: reported {reported_seed}"
            )
        row = {**summary_rows[0], "seed": job.seed}
        rows.append(row)
        for field in row:
            if field not in field_order:
                field_order.append(field)
    if not rows:
        return None, None

    destination_root = Path(output_root)
    combined_path = destination_root / "seed_results.csv"
    _write_csv_atomic(combined_path, rows, field_order)

    statistics_rows: list[dict[str, Any]] = []
    complete = len(rows) == len(job_list)
    for metric in (str(value) for value in metrics):
        values = [_finite_float(row.get(metric)) for row in rows]
        numeric = [value for value in values if value is not None]
        if not numeric:
            continue
        statistics_rows.append(
            {
                "metric": metric,
                "n": len(numeric),
                "expected_n": len(job_list),
                "complete": complete and len(numeric) == len(job_list),
                "mean": statistics.fmean(numeric),
                "std": statistics.stdev(numeric) if len(numeric) > 1 else "",
                "min": min(numeric),
                "max": max(numeric),
            }
        )
    statistics_path = destination_root / "multiseed_statistics.csv"
    _write_csv_atomic(
        statistics_path,
        statistics_rows,
        ("metric", "n", "expected_n", "complete", "mean", "std", "min", "max"),
    )
    return combined_path, statistics_path


def _run_phase(
    job: MultiSeedJob,
    phase: str,
    root: Path,
    environment: Mapping[str, str],
) -> bool:
    config_path = (
        job.train_config_path if phase == "train" else job.evaluate_config_path
    )
    command = [sys.executable, "-B", str(root / "scripts" / "heri.py"), phase, str(config_path)]
    print(f"[seed={job.seed}] {phase}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=root, env=dict(environment), check=False)
    if completed.returncode != 0:
        return False
    return (
        _training_is_complete(job)
        if phase == "train"
        else job.expected_summary_path.is_file()
    )


def _training_is_complete(job: MultiSeedJob) -> bool:
    """Accept only a completion marker, best artifact, and matching effective seed."""

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
            effective = load_yaml_config(marker.parent / "effective_config.yaml")
        except (OSError, json.JSONDecodeError, ConfigError):
            continue
        if payload.get("status") != "complete":
            continue
        if payload.get("ablation_variant") not in (None, "", "canonical"):
            continue
        try:
            project_seed = int(effective.get("project", {}).get("seed", -1))
            dataloader_seed = int(effective.get("dataloader", {}).get("seed", -1))
        except (TypeError, ValueError):
            continue
        if project_seed != job.seed:
            continue
        if dataloader_seed != job.seed:
            continue
        if (marker.parent / "checkpoints" / "best.pt").is_file():
            return True
    return False


def _validate_canonical_bases(
    train_config: Mapping[str, Any], evaluate_config: Mapping[str, Any]
) -> None:
    for label, config, mode in (
        ("training", train_config, "train"),
        ("evaluation", evaluate_config, "evaluate"),
    ):
        if config.get("model", {}).get("model_type") != "hera":
            raise ConfigError(f"The {label} base config must be canonical HERA")
        if config.get("model", {}).get("baseline_type") is not None:
            raise ConfigError(
                f"The {label} base config is B0/B1; baselines are forbidden"
            )
        if config.get("ablation", {}).get("enabled", False):
            raise ConfigError(
                f"The {label} base config is an ablation, not canonical HERA"
            )
        validate_hera_config(config, mode=mode)
    for section in _SHARED_PROTOCOL_SECTIONS:
        if train_config.get(section) != evaluate_config.get(section):
            raise ConfigError(
                f"Canonical train/evaluation configs disagree in scientific section {section!r}"
            )


def _bind_seed_protocol(
    train_config: dict[str, Any],
    evaluate_config: dict[str, Any],
    *,
    seed: int,
    train_root: str,
    evaluate_root: str,
) -> None:
    train_config["project"]["name"] = "HERA-DGM4-multiseed"
    train_config["project"]["seed"] = seed
    train_config["project"]["config_name"] = f"multiseed/seed_{seed}/train"
    train_config["project"].pop("run_id", None)
    train_config["dataloader"]["seed"] = seed
    train_config["output"]["root"] = train_root
    train_config["training"]["resume_from"] = None
    train_config["training"]["auto_resume"]["enabled"] = True
    train_config["training"]["auto_resume"]["run_dir"] = None

    evaluate_config["project"]["name"] = "HERA-DGM4-multiseed"
    evaluate_config["project"]["seed"] = seed
    evaluate_config["project"]["config_name"] = f"multiseed/seed_{seed}/evaluate"
    evaluate_config["project"]["run_id"] = f"hera_seed{seed}_test"
    evaluate_config["dataloader"]["seed"] = seed
    evaluate_config["output"]["root"] = evaluate_root
    evaluate_config["training"]["epochs"] = int(train_config["training"]["epochs"])
    evaluate_config["evaluation"]["checkpoint_path"] = None
    evaluate_config["evaluation"]["thresholds_path"] = None
    resolution = evaluate_config["evaluation"]["checkpoint_resolution"]
    resolution["enabled"] = True
    resolution["mode"] = "latest"
    resolution["output_root"] = train_root
    resolution["run_dir"] = None
    resolution["checkpoint_name"] = "best.pt"
    resolution["thresholds_name"] = "best_thresholds.json"


def _assert_seed_is_the_only_experimental_change(
    current: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    *,
    mode: str,
) -> None:
    if reference is None:
        return
    current_normalized = copy.deepcopy(dict(current))
    reference_normalized = copy.deepcopy(dict(reference))
    for config in (current_normalized, reference_normalized):
        config["project"]["seed"] = "<seed>"
        config["project"]["config_name"] = f"multiseed/<seed>/{mode}"
        if mode == "evaluate":
            config["project"]["run_id"] = "hera_seed<seed>_test"
        config["dataloader"]["seed"] = "<seed>"
        config["output"]["root"] = "<output_root>"
        if mode == "evaluate":
            config["evaluation"]["checkpoint_resolution"]["output_root"] = (
                "<train_root>"
            )
    if current_normalized != reference_normalized:
        raise ConfigError(
            f"Generated {mode} configs differ by more than seed/artifact identity"
        )


def _validated_seeds(values: Any, *, context: str) -> list[int]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
    ):
        raise ConfigError(f"{context} must be a non-empty sequence")
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ConfigError(f"{context} values must be integer random seeds")
        try:
            seed = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{context} contains a non-integer seed: {value!r}"
            ) from exc
        if seed < 0 or seed > 2**32 - 1:
            raise ConfigError(
                f"{context} seed {seed} is outside NumPy's [0, 2**32-1] range"
            )
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ConfigError(f"{context} must not contain duplicate seeds")
    return seeds


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
