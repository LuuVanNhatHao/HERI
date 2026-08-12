"""Logging setup for scripts."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure process logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
