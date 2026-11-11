"""The supported HERI commands share one parser and explicit study options."""

from pathlib import Path

import pytest

from heri.cli import build_parser
from heri.experiments.runner import run_study


def test_lifecycle_commands_parse_without_loading_models() -> None:
    parser = build_parser()
    for command in ("prepare-data", "train", "evaluate"):
        args = parser.parse_args([command, "config.yaml"])
        assert args.command == command
        assert args.config == Path("config.yaml")
    args = parser.parse_args(["reconstruct-samm", "--source-root", "official"])
    assert args.source_root == Path("official")
    assert args.output_root == Path("data/SAMM")
    args = parser.parse_args(["validate", "--section", "equations"])
    assert args.section == "equations"
    args = parser.parse_args(["audit-study", "manifest.json", "--output", "audit.json"])
    assert args.manifest == Path("manifest.json")
    assert args.output == Path("audit.json")
    for kind in ("ablation", "cycle", "multiseed", "samm", "scientific"):
        args = parser.parse_args(["study", kind, "config.yaml", "--dry-run"])
        assert args.kind == kind
        assert args.dry_run is True


@pytest.mark.parametrize(
    ("kind", "flags"),
    [
        ("ablation", ["--seeds", "7"]),
        ("cycle", ["--manifest", "manifest.json"]),
        ("multiseed", ["--variants", "full_hera"]),
        ("scientific", ["--keep-checkpoints"]),
        ("samm", ["--prune-last-after-evaluation"]),
        ("scientific", ["--resume", "--phase", "evaluate"]),
        ("cycle", ["--prune-last-after-evaluation", "--phase", "train"]),
        ("samm", ["--seeds", "7", "7"]),
    ],
)
def test_study_rejects_incompatible_options_before_reading_config(
    kind: str, flags: list[str], tmp_path: Path
) -> None:
    args = build_parser().parse_args(["study", kind, "missing.yaml", *flags])
    with pytest.raises(ValueError):
        run_study(args, tmp_path)
