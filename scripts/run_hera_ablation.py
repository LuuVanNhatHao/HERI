"""Run the controlled production HERA ablation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.ablation_runner import (
    aggregate_ablation_summaries,
    load_ablation_suite,
    materialize_ablation_jobs,
    run_ablation_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and sequentially run the six production HERA ablations. "
            "B0/B1 baselines are intentionally excluded."
        )
    )
    parser.add_argument("suite", help="Path to the ablation-suite YAML manifest")
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Optional subset of manifest variants, in manifest order",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "evaluate"),
        default="all",
        help="Run training plus evaluation, or only one phase",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate full configs without loading models or launching CUDA",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Attempt later variants after a failed subprocess",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_ablation_suite(args.suite)
    jobs = materialize_ablation_jobs(
        suite,
        repo_root=ROOT,
        selected_variants=args.variants,
    )
    print("Controlled HERA ablations:")
    for job in jobs:
        print(
            f"  {job.variant}: train={job.train_config_path} "
            f"evaluate={job.evaluate_config_path}"
        )
    if args.dry_run:
        print("Dry run complete: production configs validated; no model/GPU process was started.")
        return 0

    fail_fast = bool(suite.get("fail_fast", True)) and not args.keep_going
    run_ablation_jobs(
        jobs,
        repo_root=ROOT,
        phase=args.phase,
        environment=suite.get("environment", {}),
        fail_fast=fail_fast,
    )
    aggregate_output = Path(str(suite["output_root"])) / "ablation_summary.csv"
    if not aggregate_output.is_absolute():
        aggregate_output = ROOT / aggregate_output
    aggregate_path = aggregate_ablation_summaries(
        jobs,
        output_path=aggregate_output,
    )
    if aggregate_path is not None:
        print(f"Ablation summary: {aggregate_path}")
    elif args.phase != "train":
        print("No evaluation summary was found to aggregate.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
