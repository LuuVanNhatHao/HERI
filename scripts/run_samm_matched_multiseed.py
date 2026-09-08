"""Run the prespecified matched SAMM suite sequentially across three seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.samm_matched_multiseed import (  # noqa: E402
    aggregate_samm_matched_results,
    load_samm_matched_suite,
    materialize_samm_matched_jobs,
    run_samm_matched_jobs,
    select_samm_matched_jobs,
)
from src.utils.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run matched SAMM experiments for canonical HERA, "
            "same-backbone B1, no_router, and generic_grounding."
        )
    )
    parser.add_argument("suite", help="Path to the matched SAMM suite YAML")
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Optional recovery/debug subset; manifest execution order is preserved",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional recovery/debug subset; manifest execution order is preserved",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "evaluate"),
        default="all",
        help="Run train+test evaluation, or only one phase",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize and validate all configs without loading a model or CUDA",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Attempt later jobs after a failed job",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Retain suite best.pt and last.pt files after successful evaluation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_samm_matched_suite(args.suite)
    all_jobs = materialize_samm_matched_jobs(suite, repo_root=ROOT)
    jobs = select_samm_matched_jobs(
        all_jobs,
        variants=args.variants,
        seeds=args.seeds,
    )

    print("Matched SAMM multi-seed execution order:")
    for index, job in enumerate(jobs, start=1):
        reuse = f", reuse={job.reusable_summary_path}" if job.reusable_summary_path else ""
        print(
            f"  {index:02d}/{len(jobs):02d} seed={job.seed} variant={job.variant}: "
            f"train={job.train_config_path}, evaluate={job.evaluate_config_path}{reuse}"
        )

    effective = load_yaml_config(jobs[0].train_config_path)
    batch_size = int(effective["dataloader"]["batch_size"])
    accumulation = int(effective["training"]["gradient_accumulation_steps"])
    print(
        "Shared protocol: "
        f"seeds={suite['seeds']}, epochs={effective['training']['epochs']}, "
        f"stage_A_epochs={effective['training']['stages']['stage_a_freeze_epochs']}, "
        f"stage_B_start={effective['training']['stages']['stage_b_full_unfreeze']['start_epoch']}, "
        f"early_stop_patience={effective['training']['early_stopping']['patience']}, "
        f"effective_batch_size={batch_size * accumulation}"
    )
    if args.keep_checkpoints:
        print("Checkpoint retention override: best.pt and last.pt will be kept.")
    else:
        print(
            "Disk-safe retention: suite checkpoints are pruned only after a validated "
            "50,000-row test summary and prediction CSV."
        )
    if args.phase == "train" and not args.keep_checkpoints:
        print(
            "WARNING: --phase train cannot prune checkpoints because evaluation has not "
            "validated them; prefer the default --phase all on limited disks.",
            file=sys.stderr,
        )
    if args.dry_run:
        print(
            f"Dry run complete: {len(jobs)} configs validated; no model/GPU process started."
        )
        return 0

    fail_fast = bool(suite.get("fail_fast", True)) and not args.keep_going
    execution_error: RuntimeError | None = None
    try:
        run_samm_matched_jobs(
            jobs,
            suite=suite,
            repo_root=ROOT,
            phase=args.phase,
            fail_fast=fail_fast,
            keep_checkpoints=args.keep_checkpoints,
        )
    except RuntimeError as exc:
        execution_error = exc
        print(f"Matched SAMM suite stopped with an error: {exc}", file=sys.stderr)

    output_root = Path(str(suite["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    artifacts = aggregate_samm_matched_results(
        all_jobs,
        suite=suite,
        output_root=output_root,
    )
    if artifacts:
        for label, path in artifacts.items():
            print(f"{label}: {path}")

    if execution_error is not None:
        return 1
    selected_all = len(jobs) == len(all_jobs)
    if args.phase in {"all", "evaluate"} and selected_all:
        missing = [
            f"seed={job.seed}/{job.variant}"
            for job in all_jobs
            if not job.expected_summary_path.is_file()
        ]
        if missing:
            print(
                "Matched SAMM suite is incomplete; missing: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1
        print("Matched SAMM suite complete: all 12 test summaries are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
