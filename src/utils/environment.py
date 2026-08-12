"""Runtime environment metadata collection."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment_metadata(repo_root: str | Path = ".") -> dict[str, Any]:
    """Collect reproducibility-relevant environment metadata."""

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(Path(repo_root)),
        "packages": {
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "torch-geometric": _version("torch-geometric"),
            "transformers": _version("transformers"),
            "tokenizers": _version("tokenizers"),
            "Pillow": _version("Pillow"),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit-learn": _version("scikit-learn"),
            "PyYAML": _version("PyYAML"),
            "tqdm": _version("tqdm"),
        },
    }
    try:
        import torch

        metadata["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            metadata["cuda"].update(
                {
                    "current_device": device,
                    "gpu_name": torch.cuda.get_device_name(device),
                    "total_memory_bytes": props.total_memory,
                }
            )
    except Exception as exc:
        metadata["cuda"] = {"available": False, "error": repr(exc)}
    return metadata
