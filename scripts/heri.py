"""Repository entry point; usable without installing the package."""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heri.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(repo_root=ROOT))
