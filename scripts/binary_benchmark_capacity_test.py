"""Verify full-backbone training capacity for a binary benchmark without checkpoints."""

# ruff: noqa: E402 -- executable scripts add the repository root before src imports.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.capacity import (
    build_binary_capacity_config,
    run_binary_capacity_test,
)
from src.benchmarks.config import validate_binary_benchmark_config
from src.utils.config import load_yaml_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    base = load_yaml_config(args.config)
    config = build_binary_capacity_config(
        base,
        batch_size=args.batch_size,
        batches=args.batches,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    validate_binary_benchmark_config(config, mode="train")
    setup_logging(str(config["logging"]["level"]))
    set_seed(int(config["project"]["seed"]))
    report_path = args.report or Path("outputs/benchmark_capacity") / (
        f"{config['dataset']['adapter']}_bs{args.batch_size}_ga"
        f"{args.gradient_accumulation_steps}.json"
    )
    try:
        report = run_binary_capacity_test(config, batches=args.batches)
    except torch.OutOfMemoryError as error:
        next_batch = max(1, args.batch_size // 2)
        report = {
            "status": "oom",
            "dataset": config["dataset"]["name"],
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "error": str(error),
            "recommended_next_batch_size": next_batch,
            "recommended_accumulation_to_preserve_effective_batch": max(
                1,
                (args.batch_size * args.gradient_accumulation_steps + next_batch - 1)
                // next_batch,
            ),
            "checkpoint_written": False,
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Capacity report: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
