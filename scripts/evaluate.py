"""Evaluate Phase-2 controlled baselines from a YAML config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.hera_run import evaluate_hera
from src.evaluation.run import evaluate_baseline
from src.utils.config import load_yaml_config, validate_baseline_config, validate_hera_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/evaluate.py <config.yaml>", file=sys.stderr)
        return 2
    config = load_yaml_config(sys.argv[1])
    if config.get("model", {}).get("model_type") == "hera":
        validate_hera_config(config, mode="evaluate")
        setup_logging(str(config["logging"]["level"]))
        set_seed(int(config["project"]["seed"]))
        summary = evaluate_hera(config)
    else:
        validate_baseline_config(config, mode="evaluate")
        setup_logging(str(config["logging"]["level"]))
        set_seed(int(config["project"]["seed"]))
        summary = evaluate_baseline(config)
    print(f"Summary CSV: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
