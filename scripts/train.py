"""Train Phase-2 controlled baselines from a YAML config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.hera_trainer import HERATrainer
from src.training.trainer import BaselineTrainer
from src.utils.config import load_yaml_config, validate_baseline_config, validate_hera_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/train.py <config.yaml>", file=sys.stderr)
        return 2
    config = load_yaml_config(sys.argv[1])
    if config.get("model", {}).get("model_type") == "hera":
        validate_hera_config(config, mode="train")
        setup_logging(str(config["logging"]["level"]))
        set_seed(int(config["project"]["seed"]))
        best = HERATrainer(config).run()
    else:
        validate_baseline_config(config, mode="train")
        setup_logging(str(config["logging"]["level"]))
        set_seed(int(config["project"]["seed"]))
        best = BaselineTrainer(config).run()
    print(f"Best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
