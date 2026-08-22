"""Run canonical HERA training/evaluation for the configured random seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.multiseed_runner import (  # noqa: E402
    aggregate_multiseed_summaries,
    load_multiseed_suite,
    materialize_multiseed_jobs,
    run_multiseed_jobs,
    select_multiseed_jobs,
)
from src.utils.config import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially train and evaluate canonical HERA under independent seeds. "
            "B0, B1, and ablation variants are intentionally excluded."
        )
    )
    parser.add_argument("suite", help="Path to the multi-seed YAML manifest")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional subset of manifest seeds, executed in manifest order",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "evaluate"),
        default="all",
        help="Run training plus test evaluation, or only one phase",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize and validate configs without loading models or launching CUDA",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Attempt later seeds after a failed subprocess",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = load_multiseed_suite(args.suite)
    all_jobs = materialize_multiseed_jobs(suite, repo_root=ROOT)
    jobs = select_multiseed_jobs(all_jobs, args.seeds)

    print("Canonical HERA multi-seed jobs:")
    for job in jobs:
        print(
            f"  seed={job.seed}: train={job.train_config_path} "
            f"evaluate={job.evaluate_config_path}"
        )
    effective = load_yaml_config(jobs[0].train_config_path)
    batch_size = int(effective["dataloader"]["batch_size"])
    accumulation = int(effective["training"]["gradient_accumulation_steps"])
    print(
        "Shared protocol: "
        f"epochs={effective['training']['epochs']}, "
        f"stage_a_freeze_epochs={effective['training']['stages']['stage_a_freeze_epochs']}, "
        f"stage_b_full_unfreeze_start="
        f"{effective['training']['stages']['stage_b_full_unfreeze']['start_epoch']}, "
        f"micro_batch_size={batch_size}, gradient_accumulation_steps={accumulation}, "
        f"effective_batch_size={batch_size * accumulation}"
    )
    if args.dry_run:
        print(
            "Dry run complete: production configs validated; no model/GPU process was started."
        )
        return 0

    fail_fast = bool(suite.get("fail_fast", True)) and not args.keep_going
    run_multiseed_jobs(
        jobs,
        repo_root=ROOT,
        phase=args.phase,
        environment=suite.get("environment", {}),
        fail_fast=fail_fast,
    )

    output_root = Path(str(suite["output_root"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    combined, statistics = aggregate_multiseed_summaries(
        all_jobs,
        output_root=output_root,
        metrics=[str(value) for value in suite["aggregate_metrics"]],
    )
    if combined is not None:
        print(f"Per-seed results: {combined}")
        print(f"Mean +/- sample std: {statistics}")
    elif args.phase != "train":
        print("No per-seed evaluation summary was found.", file=sys.stderr)
        return 1

    selected_all = {job.seed for job in jobs} == {job.seed for job in all_jobs}
    if args.phase in {"all", "evaluate"} and selected_all:
        missing = [
            job.seed for job in all_jobs if not job.expected_summary_path.is_file()
        ]
        if missing:
            print(
                f"Multi-seed suite is incomplete; missing test summaries for seeds: {missing}",
                file=sys.stderr,
            )
            return 1
        print("Multi-seed suite complete: all configured seeds have test summaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
