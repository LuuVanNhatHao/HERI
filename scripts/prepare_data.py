"""Run DGM4 Phase-1 preflight from a YAML config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preflight import run_preflight
from src.utils.config import load_yaml_config, validate_prepare_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/prepare_data.py <config.yaml>", file=sys.stderr)
        return 2
    config = load_yaml_config(sys.argv[1])
    validate_prepare_config(config)
    setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    set_seed(int(config["project"]["seed"]))
    audit = run_preflight(config)
    print(f"Phase-1 preflight status: {audit['overall_status']}")
    print(f"Audit artifact: {Path(config['output']['root']) / 'dataset_audit.json'}")
    return 0 if audit["overall_status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
