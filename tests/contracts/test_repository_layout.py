from __future__ import annotations

import re

from heri.utils.config import load_yaml_config
from tests.support.paths import REPO_ROOT as ROOT


def test_config_root_is_hierarchical_and_all_yaml_resolves() -> None:
    config_root = ROOT / "config"
    assert not list(config_root.glob("*.yaml"))
    for directory in ("datasets", "experiments"):
        assert (config_root / directory).is_dir()
    configs = sorted(config_root.rglob("*.yaml"))
    assert configs
    for path in configs:
        assert isinstance(load_yaml_config(path), dict)


def test_package_and_tests_have_explicit_responsibilities() -> None:
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "src/heri/cli.py").is_file()
    assert (ROOT / "src/heri/models/heri/model.py").is_file()
    assert sorted(path.name for path in (ROOT / "scripts").glob("*.py")) == ["heri.py"]
    for directory in ("unit", "integration", "contracts", "support"):
        assert (ROOT / "tests" / directory).is_dir()
    assert not list((ROOT / "tests").glob("test_*.py"))
    assert not list((ROOT / "config").rglob("*smoke*"))


def test_readme_command_paths_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"(?:config/[^\s`]+\.yaml|scripts/[^\s`]+\.py)", readme))
    assert referenced
    for relative_path in referenced:
        assert (ROOT / relative_path).is_file(), relative_path
