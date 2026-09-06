"""Production orchestration for controlled HERA ablation suites.

The runner materializes complete train/evaluation YAML files and launches the
same canonical entry points used for the main HERA experiment.  Every variant
gets an isolated training root, so auto-resume and evaluation artifact
resolution cannot accidentally select a checkpoint from another experiment.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.utils.config import (
    ConfigError,
    dump_yaml,
    load_yaml_config,
    validate_hera_config,
)

from .ablation import HERA_ABLATION_VARIANTS, apply_hera_ablation_variant


@dataclass(frozen=True)
class AblationJob:
    """Fully materialized paths for one controlled HERA ablation."""

    variant: str
    train_config_path: Path
    evaluate_config_path: Path
    train_output_root: Path
    evaluate_output_root: Path
    expected_summary_path: Path


def load_ablation_suite(path: str | Path) -> dict[str, Any]:
    """Load and validate an ablation-suite manifest without touching a GPU."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ConfigError(f"Ablation suite does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("suite"), dict):
        raise ConfigError("Ablation manifest must contain a top-level suite mapping")
    suite = dict(manifest["suite"])
    for key in ("name", "base_train_config", "base_evaluate_config", "output_root", "variants"):
        if key not in suite:
            raise ConfigError(f"Missing mandatory ablation-suite key: suite.{key}")
    variants = suite["variants"]
    if not isinstance(variants, list) or not variants:
        raise ConfigError("suite.variants must be a non-empty list")
    if len(variants) != len(set(variants)):
        raise ConfigError("suite.variants must not contain duplicates")
    unknown = [name for name in variants if name not in HERA_ABLATION_VARIANTS]
    if unknown:
        raise ConfigError(
            f"Unknown or forbidden variants {unknown}; B0/B1 are not part of this suite. "
            f"Allowed variants: {list(HERA_ABLATION_VARIANTS)}"
        )
    return suite


def materialize_ablation_jobs(
    suite: Mapping[str, Any],
    *,
    repo_root: str | Path,
    selected_variants: Sequence[str] | None = None,
) -> list[AblationJob]:
    """Write complete production YAMLs and return executable job definitions."""

    root = Path(repo_root).resolve()
    configured_variants = [str(value) for value in suite["variants"]]
    variants = _select_variants(configured_variants, selected_variants)
    train_base_path = _resolve_path(root, suite["base_train_config"])
    evaluate_base_path = _resolve_path(root, suite["base_evaluate_config"])
    train_base = load_yaml_config(train_base_path)
    evaluate_base = load_yaml_config(evaluate_base_path)
    _validate_canonical_bases(train_base, evaluate_base)

    output_root = _resolve_path(root, suite["output_root"])
    jobs: list[AblationJob] = []
    for variant in variants:
        variant_root = output_root / variant
        config_root = variant_root / "configs"
        train_root = variant_root / "train"
        evaluate_root = variant_root / "evaluation"
        train_config = apply_hera_ablation_variant(train_base, variant)
        evaluate_config = apply_hera_ablation_variant(evaluate_base, variant)
        _bind_variant_protocol(
            train_config,
            evaluate_config,
            variant=variant,
            train_root=_portable_path(train_root, root),
            evaluate_root=_portable_path(evaluate_root, root),
            training_epochs=int(suite.get("training_epochs", train_config["training"]["epochs"])),
        )
        validate_hera_config(train_config, mode="train")
        validate_hera_config(evaluate_config, mode="evaluate")

        train_config_path = config_root / "train.yaml"
        evaluate_config_path = config_root / "evaluate.yaml"
        config_root.mkdir(parents=True, exist_ok=True)
        dump_yaml(train_config, train_config_path)
        dump_yaml(evaluate_config, evaluate_config_path)
        evaluation_run_id = str(evaluate_config["project"]["run_id"])
        jobs.append(
            AblationJob(
                variant=variant,
                train_config_path=train_config_path,
                evaluate_config_path=evaluate_config_path,
                train_output_root=train_root,
                evaluate_output_root=evaluate_root,
                expected_summary_path=evaluate_root / evaluation_run_id / "summary.csv",
            )
        )
    return jobs


def run_ablation_jobs(
    jobs: Sequence[AblationJob],
    *,
    repo_root: str | Path,
    phase: str = "all",
    environment: Mapping[str, Any] | None = None,
    fail_fast: bool = True,
) -> list[Path]:
    """Run jobs sequentially in isolated Python processes and return summaries."""

    if phase not in {"all", "train", "evaluate"}:
        raise ValueError("phase must be all, train, or evaluate")
    root = Path(repo_root).resolve()
    process_environment = os.environ.copy()
    for key, value in dict(environment or {}).items():
        process_environment[str(key)] = str(value)
    failures: list[str] = []
    for job in jobs:
        commands: list[tuple[str, Path]] = []
        if phase in {"all", "train"}:
            if _training_is_complete(job):
                print(f"[{job.variant}] train: already complete; skipping", flush=True)
            else:
                commands.append(("train", job.train_config_path))
        if phase in {"all", "evaluate"}:
            if job.expected_summary_path.is_file():
                print(f"[{job.variant}] evaluate: summary already exists; skipping", flush=True)
            else:
                commands.append(("evaluate", job.evaluate_config_path))
        for command_name, config_path in commands:
            command = [
                sys.executable,
                str(root / "scripts" / f"{command_name}.py"),
                str(config_path),
            ]
            print(f"[{job.variant}] {command_name}: {' '.join(command)}", flush=True)
            completed = subprocess.run(
                command,
                cwd=root,
                env=process_environment,
                check=False,
            )
            phase_complete = completed.returncode == 0
            if command_name == "train" and phase_complete:
                phase_complete = _training_is_complete(job)
            if phase_complete:
                continue
            failure = (
                f"variant={job.variant} phase={command_name} exited with "
                f"code {completed.returncode} or did not write its completion artifact"
            )
            failures.append(failure)
            if fail_fast:
                raise RuntimeError(failure)
            break
    if failures:
        raise RuntimeError("Ablation suite failures: " + "; ".join(failures))
    return [job.expected_summary_path for job in jobs if job.expected_summary_path.is_file()]


def _training_is_complete(job: AblationJob) -> bool:
    """Return true only for a valid completion marker and best checkpoint."""

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
        if payload.get("ablation_variant") != job.variant:
            continue
        if (marker.parent / "checkpoints" / "best.pt").is_file():
            return True
    return False


def aggregate_ablation_summaries(
    jobs: Iterable[AblationJob],
    *,
    output_path: str | Path,
) -> Path | None:
    """Combine existing per-variant summary rows in manifest order."""

    rows: list[dict[str, Any]] = []
    field_order: list[str] = ["ablation_variant"]
    for job in jobs:
        if not job.expected_summary_path.is_file():
            continue
        with job.expected_summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            summary_rows = list(reader)
            if len(summary_rows) != 1:
                raise ValueError(
                    f"Expected one summary row in {job.expected_summary_path}, got {len(summary_rows)}"
                )
            reported_variant = summary_rows[0].get("ablation_variant")
            if reported_variant not in (None, "", job.variant):
                raise ValueError(
                    f"Summary variant mismatch for {job.variant}: reported {reported_variant!r}"
                )
            row = {**summary_rows[0], "ablation_variant": job.variant}
            rows.append(row)
            for field in row:
                if field not in field_order:
                    field_order.append(field)
    if not rows:
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _select_variants(
    configured: Sequence[str], selected: Sequence[str] | None
) -> list[str]:
    if not selected:
        return list(configured)
    requested = [str(value) for value in selected]
    unknown = [name for name in requested if name not in configured]
    if unknown:
        raise ConfigError(
            f"Requested variants are not enabled by the suite manifest: {unknown}"
        )
    return [name for name in configured if name in set(requested)]


def _validate_canonical_bases(
    train_config: Mapping[str, Any], evaluate_config: Mapping[str, Any]
) -> None:
    for label, config, mode in (
        ("training", train_config, "train"),
        ("evaluation", evaluate_config, "evaluate"),
    ):
        if config.get("model", {}).get("baseline_type") is not None:
            raise ConfigError(f"The {label} base config is B0/B1; HERA ablations forbid baselines")
        if config.get("ablation", {}).get("enabled", False):
            raise ConfigError(f"The {label} base config must be canonical HERA, not an ablation")
        validate_hera_config(config, mode=mode)
    if int(train_config["project"]["seed"]) != int(evaluate_config["project"]["seed"]):
        raise ConfigError("Training and evaluation base configs must use the same project.seed")
    if int(train_config["dataloader"]["seed"]) != int(evaluate_config["dataloader"]["seed"]):
        raise ConfigError("Training and evaluation base configs must use the same dataloader.seed")


def _bind_variant_protocol(
    train_config: dict[str, Any],
    evaluate_config: dict[str, Any],
    *,
    variant: str,
    train_root: str,
    evaluate_root: str,
    training_epochs: int,
) -> None:
    if int(training_epochs) <= 0:
        raise ConfigError("suite.training_epochs must be positive")
    dataset_name = str(train_config["dataset"]["name"])
    train_config["project"]["name"] = f"HERA-{dataset_name}-ablation-{variant}"
    train_config["project"]["config_name"] = f"ablation/{variant}/train"
    train_config["project"].pop("run_id", None)
    train_config["output"]["root"] = train_root
    train_config["training"]["auto_resume"]["run_dir"] = None
    train_config["training"]["resume_from"] = None
    train_config["training"]["epochs"] = int(training_epochs)

    evaluate_config["project"]["name"] = f"HERA-{dataset_name}-ablation-{variant}"
    evaluate_config["project"]["config_name"] = f"ablation/{variant}/evaluate"
    evaluate_config["project"]["run_id"] = f"hera_{variant}_test"
    evaluate_config["output"]["root"] = evaluate_root
    evaluate_config["training"]["epochs"] = int(training_epochs)
    evaluate_config["evaluation"]["checkpoint_path"] = None
    evaluate_config["evaluation"]["thresholds_path"] = None
    resolution = evaluate_config["evaluation"]["checkpoint_resolution"]
    resolution["enabled"] = True
    resolution["mode"] = "latest"
    resolution["output_root"] = train_root
    resolution["run_dir"] = None


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
