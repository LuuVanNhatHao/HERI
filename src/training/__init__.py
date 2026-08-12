"""Training infrastructure for HERA-DGM phases.

Trainer modules import evaluation helpers, while evaluation helpers import
``src.training.precision``. Keep package-level trainer exports lazy so importing
the precision helper does not recursively import the full trainer stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BaselineTrainer", "HERATrainer"]


def __getattr__(name: str) -> Any:
    if name == "BaselineTrainer":
        from .trainer import BaselineTrainer

        return BaselineTrainer
    if name == "HERATrainer":
        from .hera_trainer import HERATrainer

        return HERATrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
