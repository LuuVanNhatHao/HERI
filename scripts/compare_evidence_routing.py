"""Compare HERA evidence routing between DGM4 and SAMM prediction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.routing_analysis import compare_evidence_routing  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="DATASET=CSV",
        help="Prediction artifact; repeat for each dataset",
    )
    parser.add_argument("--reference", default="DGM4")
    parser.add_argument("--comparison", default="SAMM")
    parser.add_argument("--output", default="outputs/routing_dgm4_vs_samm")
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"--input must be DATASET=CSV, got {value!r}")
        if name in result:
            raise ValueError(f"Duplicate dataset input: {name!r}")
        result[name] = Path(path)
    return result


def main() -> int:
    args = parse_args()
    report = compare_evidence_routing(
        inputs=_parse_inputs(args.input),
        output_dir=args.output,
        reference=args.reference,
        comparison=args.comparison,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
