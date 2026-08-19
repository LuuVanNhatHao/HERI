"""Evaluate a binary benchmark with the saved validation threshold."""

# ruff: noqa: E402 -- executable scripts add the repository root before src imports.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.config import validate_binary_benchmark_config
from src.benchmarks.evaluation import BinaryBenchmarkEvaluator
from src.utils.config import load_yaml_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    validate_binary_benchmark_config(config, mode="evaluate")
    setup_logging(str(config["logging"]["level"]))
    set_seed(int(config["project"]["seed"]))
    summary = BinaryBenchmarkEvaluator(config).run()
    print(f"Summary CSV: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
