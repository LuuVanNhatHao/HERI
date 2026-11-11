"""One command surface for the HERI reproducible experiment lifecycle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heri", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("prepare-data", "Audit annotation identity, geometry and dataset availability"),
        ("train", "Train HERI or a configured paper baseline"),
        ("evaluate", "Evaluate a checkpoint with the paper protocol"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("config", type=Path)
    sub = commands.add_parser("reconstruct-samm", help="Reconstruct frozen SAMM annotations")
    sub.add_argument("--source-root", type=Path, required=True)
    sub.add_argument("--output-root", type=Path, default=Path("data/SAMM"))
    sub = commands.add_parser(
        "validate", help="Bounded offline CPU equation and integration checks"
    )
    sub.add_argument(
        "--section", choices=("all", "equations", "runtime", "integration", "config"), default="all"
    )
    sub = commands.add_parser("study", help="Materialize or execute controlled paper experiments")
    sub.add_argument("kind", choices=("ablation", "cycle", "multiseed", "samm", "scientific"))
    sub.add_argument("config", type=Path)
    sub.add_argument(
        "--dry-run", action="store_true", help="Validate/materialize jobs without loading models"
    )
    sub.add_argument("--variants", nargs="+")
    sub.add_argument("--seeds", nargs="+", type=int)
    sub.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    sub.add_argument("--keep-going", action="store_true")
    sub.add_argument("--keep-checkpoints", action="store_true")
    sub.add_argument("--prune-last-after-evaluation", action="store_true")
    sub.add_argument("--manifest", type=Path, help="Existing scientific study manifest")
    sub.add_argument(
        "--resume", action="store_true", help="Resume a scientific job from its own last.pt"
    )
    return parser


def main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    # Apply before importing torch, pytest or transformers, including child jobs.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    args = build_parser().parse_args(argv)
    root = (repo_root or Path.cwd()).resolve()
    os.chdir(root)
    if args.command == "validate":
        from heri.validation import run_checks

        return run_checks(root, section=args.section)
    if args.command == "study":
        from heri.experiments.runner import run_study

        return run_study(args, root)
    if args.command == "reconstruct-samm":
        return _reconstruct_samm(args)
    return _run_config(args)


def _run_config(args: argparse.Namespace) -> int:
    from heri.utils.config import (
        load_yaml_config,
        validate_baseline_config,
        validate_hera_config,
        validate_prepare_config,
    )
    from heri.utils.logging import setup_logging
    from heri.utils.seed import set_seed

    config = load_yaml_config(args.config)
    if args.command == "prepare-data":
        validate_prepare_config(config)
    else:
        validator = (
            validate_hera_config
            if config.get("model", {}).get("model_type") == "hera"
            else validate_baseline_config
        )
        validator(config, mode=args.command)
    setup_logging(str(config.get("logging", {}).get("level", "INFO")))
    set_seed(int(config["project"]["seed"]))
    if args.command == "prepare-data":
        from heri.data.preflight import run_preflight

        audit = run_preflight(config)
        print(f"Preflight: {audit['overall_status']}; artifacts: {config['output']['root']}")
        return int(audit["overall_status"] == "FAIL")
    is_heri = config.get("model", {}).get("model_type") == "hera"
    if args.command == "train":
        from heri.training.heri_trainer import HERATrainer
        from heri.training.trainer import BaselineTrainer

        checkpoint = (HERATrainer if is_heri else BaselineTrainer)(config).run()
        print(f"Best checkpoint: {checkpoint}")
    else:
        from heri.evaluation.heri_run import evaluate_hera, evaluate_hera_text_grounding_diagnostic
        from heri.evaluation.run import evaluate_baseline

        diagnostic = (
            config.get("evaluation", {}).get("text_grounding_diagnostic", {}).get("enabled", False)
        )
        evaluate = (
            (evaluate_hera_text_grounding_diagnostic if diagnostic else evaluate_hera)
            if is_heri
            else evaluate_baseline
        )
        print(f"Summary CSV: {evaluate(config)}")
    return 0


def _reconstruct_samm(args: argparse.Namespace) -> int:
    from heri.data.samm import SAMM_EXPECTED_SHA256, SAMM_EXPECTED_SPLIT_ROWS
    from heri.data.samm_reconstruction import annotation_fingerprint, reconstruct_annotation_bytes

    payloads, records = {}, {}
    for split in ("train", "val", "test"):
        source = json.loads((args.source_root / f"{split}.json").read_bytes())
        if split == "train" and len(source) != 190_970:
            raise ValueError("Reconstruction requires the official 190970-row training source")
        annotations, ids = reconstruct_annotation_bytes(source, split=split)
        record = annotation_fingerprint(annotations, ids)
        if (
            record["rows"] != SAMM_EXPECTED_SPLIT_ROWS[split]
            or record["sha256"] != SAMM_EXPECTED_SHA256[split]
        ):
            raise ValueError(f"Reconstructed {split} differs from the frozen annotation bytes: {record}")
        payloads[f"{split}.json"] = annotations
        if split == "train":
            payloads["samm20k_train_ids.txt"] = ids
        records[split] = record
    manifest = {
        "protocol_id": "samm20k_seed777",
        "selection_seed": 777,
        "python": platform.python_version(),
        "splits": records,
        "exact_published_baseline_subset_identity": "not_established",
        "image_files_checked": False,
    }
    payloads["samm_reconstruction_manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    for filename, payload in payloads.items():
        destination = args.output_root / filename
        if destination.exists() and destination.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite different artifact: {destination}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        destination = args.output_root / filename
        if not destination.exists():
            with destination.open("xb") as handle:
                handle.write(payload)
    print(json.dumps(manifest, indent=2))
    return 0
