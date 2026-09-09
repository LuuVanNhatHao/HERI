from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
FROZEN_UNZIP_SOURCE_SHA256 = (
    "8c58475393918033eff6c6732e9f55135aa3f2f30a5b97387418c7618549e6f1"
)


def test_config_root_is_hierarchical_and_all_yaml_resolves() -> None:
    config_root = ROOT / "config"
    assert not list(config_root.glob("*.yaml"))
    for directory in ("datasets", "experiments", "diagnostics", "benchmarks"):
        assert (config_root / directory).is_dir()
    configs = sorted(config_root.rglob("*.yaml"))
    assert configs
    for path in configs:
        assert isinstance(load_yaml_config(path), dict)


def test_frozen_data_unzip_helpers_preserve_their_source_content() -> None:
    for name in ("data_unzip.py", "data_unzip_hera.py"):
        path = ROOT / name
        assert path.is_file()
        # Git may materialize LF or CRLF depending on checkout settings. Normalize
        # only line endings so this guard remains portable across Linux/Windows.
        source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        assert hashlib.sha256(source).hexdigest() == FROZEN_UNZIP_SOURCE_SHA256


def test_readme_command_paths_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"(?:config/[^\s`]+\.yaml|scripts/[^\s`]+\.py)", readme))
    assert referenced
    for relative_path in referenced:
        assert (ROOT / relative_path).is_file(), relative_path
