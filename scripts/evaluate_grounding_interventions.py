"""Evaluate a HERA checkpoint and predicted-box deletion in a separate run directory."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.hera_run import evaluate_hera
from src.utils.config import load_yaml_config, validate_hera_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional exploratory subset; report records the limit")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    config["project"]["run_id"] = str(config["project"].get("run_id", "evaluation")) + "_image_deletion"
    config["evaluation"]["image_deletion"] = {"enabled": True, "random_repeats": args.random_repeats, "max_batches": args.max_batches}
    validate_hera_config(config, mode="evaluate")
    print(evaluate_hera(config))
