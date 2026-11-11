"""Dispatch controlled HERI studies through one reproducible command surface.

Every study first materializes complete configurations. Dry runs stop before
model execution; normal runs use the same train/evaluate commands as individual
experiments. Aggregation retains the full manifest's completeness denominator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class _StudyPlan:
    suite: Mapping[str, Any]
    all_jobs: Sequence[Any]
    jobs: Sequence[Any]
    execute: Callable[[], Any]
    aggregate: Callable[[], Any]


def run_study(args: argparse.Namespace, root: Path) -> int:
    """Validate options, materialize jobs and optionally execute a paper study."""
    root = root.resolve()
    _validate_options(args)
    if args.kind == "scientific":
        return _run_scientific(args, root)

    plan = _build_plan(args, root)
    for index, job in enumerate(plan.jobs, start=1):
        label = _job_label(job)
        print(
            f"[{index}/{len(plan.jobs)}] {label}: "
            f"train={job.train_config_path}; evaluate={job.evaluate_config_path}",
            flush=True,
        )
    if args.dry_run:
        print(f"Dry run complete: {len(plan.jobs)} jobs validated; no model process started.")
        return 0

    failure: RuntimeError | None = None
    try:
        plan.execute()
    except RuntimeError as exc:
        failure = exc
        print(f"Study execution failed: {exc}", file=sys.stderr, flush=True)
    artifacts = plan.aggregate()
    if artifacts:
        print(f"Study aggregates: {artifacts}", flush=True)
    missing = (
        []
        if args.phase == "train"
        else [_job_label(job) for job in plan.jobs if not job.expected_summary_path.is_file()]
    )
    if missing:
        print("Missing selected test summaries: " + ", ".join(missing), file=sys.stderr)
    return int(failure is not None or bool(missing))


def _validate_options(args: argparse.Namespace) -> None:
    supported = {
        "variants": {"ablation", "cycle", "samm", "scientific"},
        "seeds": {"multiseed", "samm", "scientific"},
        "keep_checkpoints": {"samm"},
        "prune_last_after_evaluation": {"cycle"},
        "manifest": {"scientific"},
        "resume": {"scientific"},
    }
    for option, kinds in supported.items():
        value = getattr(args, option, None)
        if value and args.kind not in kinds:
            raise ValueError(f"--{option.replace('_', '-')} is unsupported for {args.kind} studies")
    for option in ("variants", "seeds"):
        values = getattr(args, option, None)
        if values and len(values) != len(set(values)):
            raise ValueError(f"--{option} must contain unique values")
    if args.resume and args.phase == "evaluate":
        raise ValueError("--resume requires a training phase")
    if args.prune_last_after_evaluation and args.phase == "train":
        raise ValueError("--prune-last-after-evaluation requires an evaluation phase")


def _select_jobs(jobs: Sequence[Any], args: argparse.Namespace) -> list[Any]:
    for option, attribute in (("variants", "variant"), ("seeds", "seed")):
        values = getattr(args, option, None)
        if values:
            configured = {getattr(job, attribute) for job in jobs}
            unknown = set(values) - configured
            if unknown:
                raise ValueError(f"Requested {option} are absent from the suite: {sorted(unknown)}")
    selected = [
        job
        for job in jobs
        if (not args.variants or job.variant in args.variants)
        and (not args.seeds or job.seed in args.seeds)
    ]
    if not selected:
        raise ValueError("The variant/seed selection contains no jobs")
    return selected


def _build_plan(args: argparse.Namespace, root: Path) -> _StudyPlan:
    if args.kind == "ablation":
        from .ablation_runner import (
            aggregate_ablation_summaries,
            load_ablation_suite,
            materialize_ablation_jobs,
            run_ablation_jobs,
        )

        suite = load_ablation_suite(args.config)
        all_jobs = materialize_ablation_jobs(suite, repo_root=root)
        aggregate = partial(
            aggregate_ablation_summaries,
            all_jobs,
            output_path=_output_root(suite, root) / "ablation_summary.csv",
        )
        run = run_ablation_jobs
        extra = {}
    elif args.kind == "cycle":
        from .cycle_study_runner import (
            aggregate_cycle_study_summaries,
            load_cycle_study_suite,
            materialize_cycle_study_jobs,
            run_cycle_study_jobs,
        )

        suite = load_cycle_study_suite(args.config)
        all_jobs = materialize_cycle_study_jobs(suite, repo_root=root)
        aggregate = partial(
            aggregate_cycle_study_summaries,
            all_jobs,
            output_path=_output_root(suite, root) / "cycle_study_summary.csv",
        )
        run = run_cycle_study_jobs
        extra = {"prune_last_after_evaluation": args.prune_last_after_evaluation}
    elif args.kind == "multiseed":
        from .multiseed_runner import (
            aggregate_multiseed_summaries,
            load_multiseed_suite,
            materialize_multiseed_jobs,
            run_multiseed_jobs,
        )

        suite = load_multiseed_suite(args.config)
        all_jobs = materialize_multiseed_jobs(suite, repo_root=root)
        aggregate = partial(
            aggregate_multiseed_summaries,
            all_jobs,
            output_root=_output_root(suite, root),
            metrics=[str(value) for value in suite["aggregate_metrics"]],
        )
        run = run_multiseed_jobs
        extra = {}
    elif args.kind == "samm":
        from .samm_matched_multiseed import (
            aggregate_samm_matched_results,
            load_samm_matched_suite,
            materialize_samm_matched_jobs,
            run_samm_matched_jobs,
        )

        suite = load_samm_matched_suite(args.config)
        all_jobs = materialize_samm_matched_jobs(suite, repo_root=root)
        aggregate = partial(
            aggregate_samm_matched_results,
            all_jobs,
            suite=suite,
            output_root=_output_root(suite, root),
        )
        run = run_samm_matched_jobs
        extra = {"suite": suite, "keep_checkpoints": args.keep_checkpoints}
    else:
        raise ValueError(f"Unknown study kind: {args.kind}")
    jobs = _select_jobs(all_jobs, args)
    common = {
        "repo_root": root,
        "phase": args.phase,
        "fail_fast": bool(suite.get("fail_fast", True)) and not args.keep_going,
    }
    if args.kind != "samm":
        common["environment"] = suite.get("environment", {})
    execute = partial(run, jobs, **common, **extra)
    return _StudyPlan(suite, all_jobs, jobs, execute, aggregate)


def _output_root(suite: Mapping[str, Any], root: Path) -> Path:
    output = Path(str(suite["output_root"]))
    return output if output.is_absolute() else root / output


def _job_label(job: Any) -> str:
    return " ".join(
        f"{attribute}={getattr(job, attribute)}"
        for attribute in ("variant", "seed")
        if hasattr(job, attribute)
    )


def _run_scientific(args: argparse.Namespace, root: Path) -> int:
    from .scientific_study import aggregate_study, materialize_study
    from heri.utils.config import load_yaml_config

    suite = load_yaml_config(args.config)["suite"]
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.manifest
        else materialize_study(suite, root)
    )
    manifest_path = Path(args.manifest or manifest["manifest_path"])
    _validate_scientific_manifest(manifest, root)
    known_variants = {job["variant"] for job in manifest["jobs"]}
    if set(args.variants or []) - known_variants or set(args.seeds or []) - set(manifest["seeds"]):
        raise ValueError("Requested variant/seed is absent from the manifest")
    jobs = [
        job
        for job in manifest["jobs"]
        if (not args.variants or job["variant"] in args.variants)
        and (not args.seeds or job["seed"] in args.seeds)
    ]
    if not jobs:
        raise ValueError("The variant/seed selection contains no scientific jobs")
    print(f"Scientific study: {len(jobs)} selected of {len(manifest['jobs'])} jobs", flush=True)
    failures = []
    if not args.dry_run:
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in suite.get("environment", {}).items()})
        for job in jobs:
            try:
                for phase in ("train", "evaluate") if args.phase == "all" else (args.phase,):
                    _run_scientific_phase(job, phase, manifest, args, root, environment)
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
                failures.append(f"{job['variant']} seed={job['seed']}: {exc}")
                if bool(suite.get("fail_fast", True)) and not args.keep_going:
                    break
    report = aggregate_study(manifest)
    report_path = manifest_path.parent / "paper_results.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "paper_results": str(report_path),
                "status": "dry_run" if args.dry_run else report["status"],
                "selected_jobs": len(jobs),
                "pending_jobs": len(report["pending_jobs"]),
            },
            indent=2,
        ),
        flush=True,
    )
    for failure in failures:
        print(failure, file=sys.stderr)
    missing = (
        not args.dry_run
        and args.phase != "train"
        and any(not Path(job["summary"]).is_file() for job in jobs)
    )
    return int(bool(failures) or bool(missing))


def _validate_scientific_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    from .scientific_study import digest, source_digest
    from heri.evaluation.protocol import EVALUATION_PROTOCOL
    from heri.utils.config import load_yaml_config, validate_baseline_config, validate_hera_config

    if manifest["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError(
            "Manifest evaluation protocol is incompatible with this HERI implementation"
        )
    if source_digest(root) != manifest["source_digest"]:
        raise ValueError("Source changed; materialize a new study instead of mixing runs")
    seen = set()
    for job in manifest["jobs"]:
        identity = (job["variant"], job["seed"])
        if identity in seen or job["seed"] not in manifest["seeds"]:
            raise ValueError(f"Invalid duplicate or unregistered scientific job: {identity}")
        seen.add(identity)
        for phase in ("train", "evaluate"):
            path = Path(job[f"{phase}_config"])
            config = load_yaml_config(path)
            if digest(config) != job[f"{phase}_config_hash"]:
                raise ValueError(f"Materialized config changed: {path}")
            expected = {
                "protocol_id": manifest["protocol_id"],
                "variant": job["variant"],
                "seed": job["seed"],
            }
            if config.get("scientific_study") != expected:
                raise ValueError(f"Config belongs to a different scientific job: {path}")
            validator = (
                validate_hera_config
                if config.get("model", {}).get("model_type") == "hera"
                else validate_baseline_config
            )
            validator(config, mode=phase)


def _run_scientific_phase(
    job: Mapping[str, Any],
    phase: str,
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
    root: Path,
    environment: Mapping[str, str],
) -> None:
    from .scientific_study import digest
    from heri.utils.config import dump_yaml, load_yaml_config

    path = Path(job[f"{phase}_config"])
    config = load_yaml_config(path)
    if digest(config) != job[f"{phase}_config_hash"]:
        raise ValueError(f"Materialized config changed: {path}")
    marker = path.parent / f"{phase}_completed.json"
    completion = {
        "protocol_id": manifest["protocol_id"],
        "config_hash": job[f"{phase}_config_hash"],
    }
    run_root = path.parent.parent / "train"
    artifact = Path(job["summary"]) if phase == "evaluate" else run_root / "checkpoints/best.pt"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != completion:
            raise ValueError(f"Invalid completion marker: {marker}")
        if not artifact.is_file():
            raise FileNotFoundError(f"Completed job lost artifact: {artifact}")
        return
    if phase == "train" and run_root.exists():
        last = run_root / "checkpoints/last.pt"
        if not args.resume or not last.is_file():
            raise ValueError(
                f"Existing incomplete run: {run_root}. Use --resume with its own last.pt or a new study output_root"
            )
        config["training"]["resume_from"] = str(last)
        path = path.parent / "resume.yaml"
        dump_yaml(config, path)
    command = [sys.executable, "-B", str(root / "scripts/heri.py"), phase, str(path)]
    print(f"{job['variant']} seed={job['seed']} {phase}", flush=True)
    subprocess.run(command, cwd=root, env=dict(environment), check=True)
    if not artifact.is_file():
        raise FileNotFoundError(f"Command exited without expected artifact: {artifact}")
    marker.write_text(json.dumps(completion), encoding="utf-8")
