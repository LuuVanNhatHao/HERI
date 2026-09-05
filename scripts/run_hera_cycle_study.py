"""Run HERA's controlled cycle-depth and graph-rebuild study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.cycle_study import cycle_study_protocol
from src.experiments.cycle_study_runner import (
    aggregate_cycle_study_summaries,
    load_cycle_study_suite,
    materialize_cycle_study_jobs,
    run_cycle_study_jobs,
    select_cycle_study_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched single/static/dynamic HERA graph-cycle controls"
    )
    parser.add_argument("suite", help="Path to the cycle-study YAML manifest")
    parser.add_argument("--variants", nargs="+", help="Optional variant subset")
    parser.add_argument(
        "--phase", choices=("all", "train", "evaluate"), default="all"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize and validate configs without loading a model or CUDA",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--prune-last-after-evaluation",
        action="store_true",
        help="After a valid summary exists, delete last.pt but preserve best.pt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_cycle_study_suite(args.suite)
    all_jobs = materialize_cycle_study_jobs(suite, repo_root=ROOT)
    jobs = select_cycle_study_jobs(all_jobs, args.variants)
    print("Controlled HERA cycle-depth study:")
    for job in jobs:
        protocol = cycle_study_protocol(job.variant)
        print(
            f"  {job.variant}: cycles={protocol['cycles']} "
            f"rebuild={protocol['rebuild_mode']} train={job.train_config_path} "
            f"evaluate={job.evaluate_config_path}"
        )
    if args.dry_run:
        print("Dry run complete: full production configs validated; CUDA was not started.")
        return 0

    run_cycle_study_jobs(
        jobs,
        repo_root=ROOT,
        phase=args.phase,
        environment=suite.get("environment", {}),
        fail_fast=bool(suite.get("fail_fast", True)) and not args.keep_going,
        prune_last_after_evaluation=args.prune_last_after_evaluation,
    )
    aggregate = aggregate_cycle_study_summaries(
        all_jobs,
        output_path=ROOT / str(suite["output_root"]) / "cycle_study_summary.csv",
    )
    if aggregate is not None:
        print(f"Cycle-study summary: {aggregate}")
    elif args.phase != "train":
        print("No cycle-study evaluation summaries were found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
